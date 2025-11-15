#!/usr/bin/env python3
"""
llm_engine.py

GPU-оптимизированный LLM-движок для извлечения сущностей из сценариев.
Поддерживает:
- Многоэтапную обработку (детекция сцен -> извлечение сущностей)
- Батчевую обработку для GPU
- Динамическую генерацию промптов на основе конфигурации
- Использование vllm или transformers с flash-attention
"""

import json
import re
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch
import gc

logger = logging.getLogger(__name__)

# Попытка импорта vllm
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vllm не установлен. Используется transformers.")

# Попытка импорта transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.error("transformers не установлен. Установите: pip install transformers torch")


@dataclass
class SceneDetection:
    """Результат детекции сцены"""
    scene_number: str
    scene_title: str
    scene_text: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class EntityExtraction:
    """Результат извлечения сущностей из сцены"""
    scene_number: str
    entities: Dict[str, Any]
    confidence: float


class LLMEngine:
    """GPU-оптимизированный LLM-движок для обработки сценариев"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Инициализация движка с конфигурацией"""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.use_vllm = self.config.get('llm', {}).get('use_vllm', False) and VLLM_AVAILABLE
        self.device = self.config.get('llm', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.entity_types = self.config.get('entity_types', {})
        self.presets = self.config.get('presets', {})
        
        # Проверка доступности GPU
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA недоступна, переключаемся на CPU")
            self.device = 'cpu'
        
        self._load_model()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML"""
        config_file = Path(config_path)
        
        # Если путь относительный, проверяем относительно корня проекта
        if not config_file.is_absolute() and not config_file.exists():
            # Пробуем найти в корне проекта
            root_config = Path(__file__).parent.parent / config_path
            if root_config.exists():
                config_file = root_config
            else:
                # Пробуем в текущей директории
                current_config = Path.cwd() / config_path
                if current_config.exists():
                    config_file = current_config
        
        if not config_file.exists():
            logger.warning(f"Конфиг {config_path} не найден, используем значения по умолчанию")
            return self._default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config is None:
                    logger.warning("Конфиг пуст, используем значения по умолчанию")
                    return self._default_config()
                return config
        except Exception as e:
            logger.warning(f"Ошибка чтения конфига: {e}, используем значения по умолчанию")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            'llm': {
                'model_name': 'mistralai/Mistral-7B-Instruct-v0.2',
                'use_vllm': False,
                'device': 'cuda',
                'generation': {
                    'max_new_tokens': 1024,
                    'temperature': 0.1,
                    'top_p': 0.95,
                    'top_k': 40
                },
                'batch': {
                    'scene_detection_batch_size': 10,
                    'entity_extraction_batch_size': 8
                }
            },
            'entity_types': {},
            'presets': {}
        }
    
    def _load_model(self):
        """Загрузка модели (vllm или transformers)"""
        model_name = self.config.get('llm', {}).get('model_name', 'mistralai/Mistral-7B-Instruct-v0.2')
        
        if self.use_vllm and VLLM_AVAILABLE:
            logger.info(f"Загрузка модели {model_name} через vllm...")
            try:
                self.model = LLM(
                    model=model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.9,
                    trust_remote_code=True
                )
                logger.info("Модель загружена через vllm")
                return
            except Exception as e:
                logger.warning(f"Ошибка загрузки через vllm: {e}. Переключаемся на transformers.")
                self.use_vllm = False
        
        if TRANSFORMERS_AVAILABLE:
            logger.info(f"Загрузка модели {model_name} через transformers...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Настройка для GPU с оптимизацией памяти
                if self.device == 'cuda':
                    # Используем 4-bit quantization для экономии памяти (если доступно)
                    try:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            quantization_config=quantization_config,
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить с quantization: {e}. Пробуем без quantization.")
                        # Fallback без quantization
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True
                        )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    self.model = self.model.to(self.device)
                
                # Установка pad_token если отсутствует
                if self.tokenizer.pad_token is None:
                    # Для Llama 3 используем специальный pad token
                    if 'llama' in model_name.lower() or 'meta-llama' in model_name.lower():
                        self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
                        if hasattr(self.model, 'resize_token_embeddings'):
                            self.model.resize_token_embeddings(len(self.tokenizer))
                    else:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Модель загружена через transformers")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise
        else:
            raise RuntimeError("Ни vllm, ни transformers не доступны. Установите хотя бы один.")
    
    def _generate_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Генерация промпта в формате модели"""
        model_name = self.config.get('llm', {}).get('model_name', '').lower()
        
        # Определяем формат промпта в зависимости от модели
        if 'llama' in model_name or 'meta-llama' in model_name:
            # Формат для Llama 3
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif 'mistral' in model_name or 'mixtral' in model_name:
            # Формат для Mistral/Mixtral Instruct
            return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        else:
            # Универсальный формат
            return f"{system_prompt}\n\n{user_prompt}\n\n"
    
    def _generate_scene_detection_prompt(self, screenplay_text: str) -> str:
        """Генерация промпта для детекции сцен (НЕ ИСПОЛЬЗУЕТСЯ - детекция через regex)"""
        system_prompt = """Ты - эксперт по анализу киносценариев. Твоя задача - точно разделить сценарий на отдельные сцены.

Каждая сцена начинается с заголовка, который содержит:
- Номер сцены (может быть в формате "1", "1.1", "СЦЕНА 1" и т.д.)
- Тип сцены (ИНТ./ЭКСТ./НАТ. или INT./EXT.)
- Локацию
- Время суток (ДЕНЬ/НОЧЬ/УТРО/ВЕЧЕР)

Ты должен вернуть JSON-массив, где каждый элемент - это сцена:
[
  {
    "scene_number": "номер сцены",
    "scene_title": "полный заголовок сцены (ИНТ. ЛОКАЦИЯ - ВРЕМЯ)",
    "start_pos": позиция начала в тексте (символ),
    "end_pos": позиция конца в тексте (символ)
  }
]

ВАЖНО: Отвечай ТОЛЬКО валидным JSON, без дополнительного текста до или после."""

        user_prompt = f"""Раздели следующий сценарий на сцены и верни JSON-массив:

{screenplay_text[:10000]}

Верни JSON-массив сценариев:"""

        return self._generate_prompt(system_prompt, user_prompt)
    
    def _generate_entity_extraction_prompt(self, scene_text: str, scene_number: str, entities_to_extract: List[str]) -> str:
        """Генерация промпта для извлечения сущностей на основе конфигурации"""
        
        # Формируем описание полей на основе конфига
        entity_descriptions = []
        json_schema = {}
        
        for entity_name in entities_to_extract:
            if entity_name in self.entity_types:
                entity_config = self.entity_types[entity_name]
                desc = entity_config.get('description', entity_name)
                entity_type = entity_config.get('type', 'string')
                
                entity_descriptions.append(f"- {entity_name} ({entity_type}): {desc}")
                
                # Формируем схему JSON
                if entity_type == 'list':
                    json_schema[entity_name] = []
                elif entity_type == 'boolean':
                    json_schema[entity_name] = False
                elif entity_type == 'integer':
                    json_schema[entity_name] = 0
                else:
                    json_schema[entity_name] = ""
        
        system_prompt = f"""Ты - профессиональный ассистент продюсера, анализирующий киносценарии для создания календарно-постановочного плана (КПП).
Твоя задача - точно извлечь информацию о производственных требованиях сцены.

Ты должен извлечь следующие поля:
{chr(10).join(entity_descriptions)}

ПРИМЕРЫ ПРАВИЛЬНОГО ИЗВЛЕЧЕНИЯ:

1. ЛОКАЦИЯ (location):
   ✅ ПРАВИЛЬНО: "ИНТ. КВАРТИРА - ДЕНЬ" → location = "квартира"
   ✅ ПРАВИЛЬНО: "ЭКСТ. УЛИЦА - НОЧЬ" → location = "улица"
   ✅ ПРАВИЛЬНО: "В бассейне плавают дети" → location = "бассейн"
   ❌ НЕПРАВИЛЬНО: location = "отделаются дети" (это действие, не локация!)
   ❌ НЕПРАВИЛЬНО: location = "плавают" (это глагол, не локация!)
   ВАЖНО: Локация - это МЕСТО (квартира, улица, офис, бассейн), НЕ действие или глагол!

2. ПЕРСОНАЖИ (characters):
   ✅ ПРАВИЛЬНО: Если в тексте "Иван говорит с Марией" → characters = ["Иван", "Мария"]
   ✅ ПРАВИЛЬНО: "В бассейне плавают дети" → characters = ["дети"] (если это главные персонажи)
   ❌ НЕПРАВИЛЬНО: characters = ["толпа", "массовка", "прохожие"] (это массовка, не персонажи!)
   ❌ НЕПРАВИЛЬНО: characters = ["плавают", "идут"] (это глаголы, не имена!)
   ВАЖНО: Персонажи - это имена людей или роли (Иван, Мария, режиссер), НЕ массовка и НЕ действия!

3. МАССОВКА (crowd, crowd_count):
   ✅ ПРАВИЛЬНО: "толпа из 20 человек" → crowd = "толпа прохожих", crowd_count = 20
   ✅ ПРАВИЛЬНО: "массовка: 15 студентов" → crowd = "студенты", crowd_count = 15
   ✅ ПРАВИЛЬНО: "официанты (5 чел.)" → crowd = "официанты", crowd_count = 5
   ❌ НЕПРАВИЛЬНО: crowd_count = "20 человек" (должно быть число 20, без текста!)
   ВАЖНО: crowd_count - это ТОЛЬКО число (0, 5, 20, 100), без слов "человек", "чел." и т.д.!

4. СПЕЦОБОРУДОВАНИЕ (special_equipment):
   ✅ ПРАВИЛЬНО: ["кран для камеры", "операторская тележка", "стабилизатор", "дрон"]
   ✅ ПРАВИЛЬНО: ["микрофон на журавле", "специальное освещение"]
   ❌ НЕПРАВИЛЬНО: ["стол", "стул", "телефон"] (это реквизит, не оборудование!)
   ❌ НЕПРАВИЛЬНО: ["костюм", "грим"] (это не оборудование!)
   ВАЖНО: Только профессиональное СЪЕМОЧНОЕ оборудование (кран, дрон, стабилизатор, тележка, журавль)!

5. КОСТЮМЫ (costumes):
   ✅ ПРАВИЛЬНО: ["деловой костюм", "вечернее платье", "форма полицейского"]
   ✅ ПРАВИЛЬНО: ["спортивная одежда", "военная форма"]
   ❌ НЕПРАВИЛЬНО: ["одежда"] (слишком общее, нужны детали)
   ❌ НЕПРАВИЛЬНО: ["стандартная одежда"] (если не описано, верни пустой список)
   ВАЖНО: Только если упомянуты КОНКРЕТНЫЕ детали костюма. Если костюмы стандартные - пустой список!

Отвечай ТОЛЬКО в формате JSON, строго следуя этой структуре:
{json.dumps(json_schema, ensure_ascii=False, indent=2)}

КРИТИЧЕСКИ ВАЖНО:
- Анализируй текст внимательно, не придумывай информацию, которой нет в сцене
- ЛОКАЦИЯ = место (квартира, улица), НЕ действие (плавают, идут, отделаются)!
- ПЕРСОНАЖИ = имена людей, НЕ массовка и НЕ глаголы!
- CROWD_COUNT = только число (20), НЕ текст ("20 человек")!
- SPECIAL_EQUIPMENT = только съемочное оборудование, НЕ реквизит!
- Если информация не найдена, используй пустые значения (пустая строка "", пустой список [], false для boolean, 0 для integer)
- Не добавляй поля, которых нет в схеме
- Не добавляй комментарии или пояснения
- JSON должен быть валидным и парситься без ошибок"""

        user_prompt = f"""Извлеки информацию из следующей сцены:

НОМЕР СЦЕНЫ: {scene_number}

ТЕКСТ СЦЕНЫ:
{scene_text}

ПОМНИ:
- Локация - это МЕСТО (квартира, улица, бассейн), НЕ действие!
- Персонажи - это ИМЕНА людей, НЕ массовка и НЕ глаголы!
- crowd_count - это ТОЛЬКО число, без текста!
- special_equipment - только съемочное оборудование, НЕ реквизит!

Верни JSON с извлеченными данными:"""

        return self._generate_prompt(system_prompt, user_prompt)
    
    def _call_model(self, prompts: List[str]) -> List[str]:
        """Вызов модели с батчевой обработкой"""
        if not self.model:
            raise RuntimeError("Модель не загружена")
        
        if self.use_vllm:
            # Использование vllm для батчевой обработки
            gen_params = self.config.get('llm', {}).get('generation', {})
            # Определяем stop tokens в зависимости от модели
            model_name = self.config.get('llm', {}).get('model_name', '').lower()
            if 'mistral' in model_name or 'mixtral' in model_name:
                stop_tokens = ["</s>", "[INST]", "[/INST]", "\n\n\n"]
            elif 'llama' in model_name or 'meta-llama' in model_name:
                stop_tokens = ["<|eot_id|>", "</s>", "\n\n\n"]
            else:
                stop_tokens = ["</s>", "\n\n\n"]
            
            sampling_params = SamplingParams(
                max_tokens=gen_params.get('max_new_tokens', 2048),
                temperature=gen_params.get('temperature', 0.05),
                top_p=gen_params.get('top_p', 0.9),
                top_k=gen_params.get('top_k', 40),
                stop=stop_tokens
            )
            
            outputs = self.model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        else:
            # Использование transformers
            gen_params = self.config.get('llm', {}).get('generation', {})
            
            # Токенизация батча
            logger.info(f"Токенизация {len(prompts)} промптов...")
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048  # Уменьшено для экономии памяти
            ).to(self.device)
            
            logger.info(f"Размер батча: {inputs['input_ids'].shape}, начинаю генерацию...")
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(gen_params.get('max_new_tokens', 2048), 1024),  # Ограничено до 1024
                    temperature=gen_params.get('temperature', 0.05),
                    top_p=gen_params.get('top_p', 0.9),
                    top_k=gen_params.get('top_k', 40),
                    do_sample=gen_params.get('do_sample', True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            logger.info(f"Генерация завершена, декодирую ответы...")
            
            # Декодирование
            responses = []
            for i, output in enumerate(outputs):
                # Убираем входной промпт из ответа
                input_length = inputs['input_ids'][i].shape[0]
                generated_text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                )
                responses.append(generated_text.strip())
            
            return responses
    
    def _extract_json(self, text: str) -> Optional[Any]:
        """Извлечение JSON из текста ответа модели (может вернуть dict или list)"""
        # Находим первую открывающую скобку
        start_idx = -1
        for i, char in enumerate(text):
            if char in ['{', '[']:
                start_idx = i
                break
        
        if start_idx == -1:
            return None
        
        # Балансируем скобки для правильного извлечения
        bracket = text[start_idx]
        closing = '}' if bracket == '{' else ']'
        stack = 1
        end_idx = start_idx + 1
        
        while end_idx < len(text) and stack > 0:
            if text[end_idx] == bracket:
                stack += 1
            elif text[end_idx] == closing:
                stack -= 1
            elif text[end_idx] == '"':
                # Пропускаем строки
                end_idx += 1
                while end_idx < len(text) and text[end_idx] != '"':
                    if text[end_idx] == '\\':
                        end_idx += 1  # Пропускаем экранированные символы
                    end_idx += 1
            end_idx += 1
        
        if stack == 0:
            json_str = text[start_idx:end_idx]
        else:
            # Если не удалось найти закрывающую скобку, пробуем найти вручную
            json_str = text[start_idx:]
        
        # Очистка и исправление JSON
        try:
            # Убираем trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            # Исправляем одинарные кавычки в ключах и значениях (но не внутри строк)
            # Это сложно, поэтому сначала пробуем как есть
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Пробуем заменить одинарные кавычки на двойные (осторожно)
                # Заменяем только те, что не внутри уже двойных кавычек
                json_str_fixed = json_str
                # Простая замена одинарных кавычек на двойные для ключей
                json_str_fixed = re.sub(r"'(\w+)':", r'"\1":', json_str_fixed)
                # Заменяем одинарные кавычки в значениях строк
                json_str_fixed = re.sub(r":\s*'([^']*)'", r': "\1"', json_str_fixed)
                return json.loads(json_str_fixed)
        except json.JSONDecodeError as e:
            # Последняя попытка - используем более агрессивную очистку
            try:
                # Удаляем комментарии
                json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                # Исправляем незакрытые строки
                json_str = re.sub(r'"([^"]*)$', r'"\1"', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e2:
                logger.warning(f"Не удалось распарсить JSON: {e2}. Начало текста: {json_str[:300]}")
                return None
    
    def detect_scenes(self, screenplay_text: str) -> List[SceneDetection]:
        """Шаг 1: Детекция сцен через регулярные выражения (надежнее чем LLM)"""
        logger.info("Начинаю детекцию сцен через регулярные выражения...")
        
        # Используем regex-метод как основной (более надежный)
        all_scenes = self._fallback_scene_detection(screenplay_text)
        
        logger.info(f"Обнаружено {len(all_scenes)} сцен через regex")
        return all_scenes
    
    def _fallback_scene_detection(self, text: str) -> List[SceneDetection]:
        """Основной метод детекции сцен через regex (надежнее чем LLM)"""
        scenes = []
        
        # Улучшенный паттерн для детекции сцен
        # Поддерживает различные форматы: "1. ИНТ. КВАРТИРА - ДЕНЬ", "СЦЕНА 1", "INT. LOCATION", и т.д.
        patterns = [
            # Основной паттерн: номер + тип + локация + время
            re.compile(
                r'(?P<number>\d+[-.]?\d*\.? )?\s*'
                r'(?P<type>INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.|INTERIOR\.|EXTERIOR\.)\s*'
                r'(?P<location>[^.\n\-–—]+?)(?:\.\s*(?P<sublocation>[^.\n\-–—]+?))?\s*[.\-\s–—]*\s*'
                r'(?P<time>ДЕНЬ|НОЧЬ|УТРО|ВЕЧЕР|РАССВЕТ|ЗАКАТ|DAY|NIGHT|MORNING|EVENING|DAWN|DUSK)?',
                re.MULTILINE | re.IGNORECASE
            ),
            # Альтернативный паттерн: "СЦЕНА 1" или "SCENE 1"
            re.compile(
                r'(?:СЦЕНА|SCENE)\s*(?P<number>\d+[-.]?\d*)\s*[:\-]?\s*'
                r'(?P<type>INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.)?\s*'
                r'(?P<location>[^\n]+?)(?:\s*[-–—]\s*(?P<time>ДЕНЬ|НОЧЬ|УТРО|ВЕЧЕР))?',
                re.MULTILINE | re.IGNORECASE
            ),
            # Простой паттерн: только тип + локация
            re.compile(
                r'^(?P<type>INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.)\s*'
                r'(?P<location>[^.\n\-–—]+?)(?:\s*[-–—]\s*(?P<time>ДЕНЬ|НОЧЬ|УТРО|ВЕЧЕР))?',
                re.MULTILINE | re.IGNORECASE
            )
        ]
        
        # Собираем все совпадения
        all_matches = []
        for pattern in patterns:
            matches = list(pattern.finditer(text))
            for match in matches:
                all_matches.append((match.start(), match))
        
        # Сортируем по позиции в тексте
        all_matches.sort(key=lambda x: x[0])
        
        # Удаляем дубликаты (если несколько паттернов нашли одно и то же)
        unique_matches = []
        seen_positions = set()
        for pos, match in all_matches:
            # Проверяем, что это не дубликат (в пределах 50 символов)
            is_duplicate = False
            for seen_pos in seen_positions:
                if abs(pos - seen_pos) < 50:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_matches.append((pos, match))
                seen_positions.add(pos)
        
        # Создаем сцены
        for i, (start, match) in enumerate(unique_matches):
            # Определяем конец сцены (начало следующей или конец текста)
            if i + 1 < len(unique_matches):
                end = unique_matches[i + 1][0]
            else:
                end = len(text)
            
            # Извлекаем номер сцены
            scene_number = match.group('number')
            if not scene_number:
                scene_number = str(i + 1)
            else:
                scene_number = scene_number.strip().rstrip('.')
            
            # Извлекаем тип сцены
            scene_type = match.group('type') or ''
            scene_type = scene_type.strip().rstrip('.')
            
            # Извлекаем локацию
            location = match.group('location') or ''
            location = location.strip()
            
            # Извлекаем подобъект
            sublocation = match.group('sublocation') or ''
            sublocation = sublocation.strip()
            
            # Извлекаем время
            time = match.group('time') or ''
            time = time.strip()
            
            # Формируем заголовок сцены
            scene_title_parts = []
            if scene_type:
                scene_title_parts.append(scene_type.upper())
            if location:
                scene_title_parts.append(location)
            if sublocation:
                scene_title_parts.append(sublocation)
            if time:
                scene_title_parts.append(time.upper())
            
            scene_title = ' '.join(scene_title_parts) if scene_title_parts else match.group(0)
            
            # Извлекаем текст сцены (от начала заголовка до начала следующей сцены)
            scene_text = text[start:end].strip()
            
            # Если текст сцены не содержит заголовок, добавляем его
            if scene_title and scene_title not in scene_text[:100]:
                scene_text = f"{scene_title}\n\n{scene_text}"
            
            scene = SceneDetection(
                scene_number=scene_number,
                scene_title=scene_title,
                scene_text=scene_text,
                start_pos=start,
                end_pos=end,
                confidence=0.9  # Высокая уверенность для regex-метода
            )
            scenes.append(scene)
        
        # Если не нашли сцены через паттерны, пробуем разбить по двойным переводам строки
        if not scenes:
            logger.warning("Не найдено сцен через паттерны, пробуем разбить по параграфам...")
            paragraphs = re.split(r'\n{3,}', text)
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if len(para) >= 50:  # Минимальная длина сцены
                    scene = SceneDetection(
                        scene_number=str(i + 1),
                        scene_title=f"Сцена {i + 1}",
                        scene_text=para,
                        start_pos=text.find(para),
                        end_pos=text.find(para) + len(para),
                        confidence=0.5
                    )
                    scenes.append(scene)
        
        return scenes
    
    def _extract_scene_text_by_title(self, text: str, title: str) -> str:
        """Извлечение текста сцены по заголовку"""
        # Ищем заголовок в тексте
        title_escaped = re.escape(title)
        match = re.search(title_escaped, text, re.IGNORECASE)
        if match:
            start = match.end()
            # Ищем следующий заголовок сцены
            next_match = re.search(
                r'(?:INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.)',
                text[start:],
                re.IGNORECASE
            )
            end = start + next_match.start() if next_match else len(text)
            return text[start:end].strip()
        return ""
    
    def extract_entities_batch(
        self,
        scenes: List[SceneDetection],
        preset: str = "full",
        custom_entities: Optional[List[str]] = None
    ) -> List[EntityExtraction]:
        """Шаг 2: Извлечение сущностей батчами"""
        logger.info(f"Начинаю извлечение сущностей для {len(scenes)} сцен...")
        
        # Определяем список сущностей для извлечения
        if custom_entities:
            entities_to_extract = custom_entities
        elif preset in self.presets:
            entities_to_extract = self.presets[preset].get('entities', [])
        else:
            # По умолчанию используем все доступные
            entities_to_extract = list(self.entity_types.keys())
        
        batch_size = self.config.get('llm', {}).get('batch', {}).get('entity_extraction_batch_size', 8)
        results = []
        
        # Обрабатываем батчами
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i:i + batch_size]
            logger.info(f"Обработка батча {i//batch_size + 1}/{(len(scenes) + batch_size - 1)//batch_size}")
            
            # Генерируем промпты для батча
            prompts = []
            for scene in batch:
                # Ограничиваем длину текста сцены для экономии памяти
                scene_text_limited = scene.scene_text[:2000] if len(scene.scene_text) > 2000 else scene.scene_text
                prompt = self._generate_entity_extraction_prompt(
                    scene_text_limited,
                    scene.scene_number,
                    entities_to_extract
                )
                prompts.append(prompt)
            
            # Вызываем модель батчем
            logger.info(f"Вызываю модель для батча из {len(prompts)} сцен...")
            import time
            start_time = time.time()
            responses = self._call_model(prompts)
            elapsed = time.time() - start_time
            logger.info(f"Модель обработала батч за {elapsed:.1f} секунд")
            
            # Парсим результаты
            for scene, response in zip(batch, responses):
                entities = self._extract_json(response)
                
                # Убеждаемся, что entities - это словарь
                if not isinstance(entities, dict):
                    if isinstance(entities, list):
                        logger.warning(f"Сцена {scene.scene_number}: получен список вместо словаря. Используем пустой словарь.")
                    entities = {}
                
                # Валидация и очистка результатов
                try:
                    try:
                        from app.result_validator import ResultValidator
                    except ImportError:
                        from result_validator import ResultValidator
                    validator = ResultValidator()
                    entities = validator.validate_and_clean(entities, scene.scene_text)
                except ImportError:
                    logger.warning("ResultValidator не найден, пропускаем валидацию")
                except Exception as e:
                    logger.warning(f"Ошибка валидации результатов: {e}")
                
                extraction = EntityExtraction(
                    scene_number=scene.scene_number,
                    entities=entities,
                    confidence=0.8 if entities else 0.5
                )
                results.append(extraction)
            
            # Очистка памяти
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"Извлечение сущностей завершено для {len(results)} сцен")
        return results
    
    def process_screenplay(
        self,
        screenplay_text: str,
        preset: str = "full",
        custom_entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Полный пайплайн обработки сценария"""
        # Шаг 1: Детекция сцен
        scenes = self.detect_scenes(screenplay_text)
        
        if not scenes:
            logger.warning("Сцены не обнаружены")
            return []
        
        # Шаг 2: Извлечение сущностей
        extractions = self.extract_entities_batch(scenes, preset, custom_entities)
        
        # Объединяем результаты
        results = []
        for scene, extraction in zip(scenes, extractions):
            # Проверяем, что entities - это словарь, а не список
            if isinstance(extraction.entities, dict):
                result = {
                    'scene_number': scene.scene_number,
                    'scene_title': scene.scene_title,
                    'scene_text': scene.scene_text,
                    **extraction.entities
                }
            elif isinstance(extraction.entities, list):
                # Если это список, создаем словарь с базовыми полями
                logger.warning(f"Сцена {scene.scene_number}: entities это список, а не словарь. Используем пустой словарь.")
                result = {
                    'scene_number': scene.scene_number,
                    'scene_title': scene.scene_title,
                    'scene_text': scene.scene_text
                }
            else:
                # Fallback
                result = {
                    'scene_number': scene.scene_number,
                    'scene_title': scene.scene_title,
                    'scene_text': scene.scene_text
                }
            results.append(result)
        
        return results
    
    def __del__(self):
        """Освобождение ресурсов"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

