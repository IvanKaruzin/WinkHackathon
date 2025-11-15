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
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("Модель загружена через transformers")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                raise
        else:
            raise RuntimeError("Ни vllm, ни transformers не доступны. Установите хотя бы один.")
    
    def _generate_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Генерация промпта в формате модели"""
        # Формат для Mistral Instruct
        return f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    
    def _generate_scene_detection_prompt(self, screenplay_text: str) -> str:
        """Генерация промпта для детекции сцен"""
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
        
        system_prompt = f"""Ты - ассистент продюсера, анализирующий сценарии для кинопроизводства.
Твоя задача - извлечь точную информацию о производственных требованиях сцены.

Ты должен извлечь следующие поля:
{chr(10).join(entity_descriptions)}

Отвечай ТОЛЬКО в формате JSON, строго следуя этой структуре:
{json.dumps(json_schema, ensure_ascii=False, indent=2)}

ВАЖНО:
- Если информация не найдена, используй пустые значения (пустая строка "", пустой список [], false для boolean, 0 для integer)
- Не добавляй поля, которых нет в схеме
- Не добавляй комментарии или пояснения
- JSON должен быть валидным и парситься без ошибок"""

        user_prompt = f"""Извлеки информацию из следующей сцены:

НОМЕР СЦЕНЫ: {scene_number}

ТЕКСТ СЦЕНЫ:
{scene_text}

Верни JSON с извлеченными данными:"""

        return self._generate_prompt(system_prompt, user_prompt)
    
    def _call_model(self, prompts: List[str]) -> List[str]:
        """Вызов модели с батчевой обработкой"""
        if not self.model:
            raise RuntimeError("Модель не загружена")
        
        if self.use_vllm:
            # Использование vllm для батчевой обработки
            gen_params = self.config.get('llm', {}).get('generation', {})
            sampling_params = SamplingParams(
                max_tokens=gen_params.get('max_new_tokens', 1024),
                temperature=gen_params.get('temperature', 0.1),
                top_p=gen_params.get('top_p', 0.95),
                top_k=gen_params.get('top_k', 40),
                stop=["</s>", "\n\n\n"]
            )
            
            outputs = self.model.generate(prompts, sampling_params)
            return [output.outputs[0].text.strip() for output in outputs]
        else:
            # Использование transformers
            gen_params = self.config.get('llm', {}).get('generation', {})
            
            # Токенизация батча
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            ).to(self.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_params.get('max_new_tokens', 1024),
                    temperature=gen_params.get('temperature', 0.1),
                    top_p=gen_params.get('top_p', 0.95),
                    top_k=gen_params.get('top_k', 40),
                    do_sample=gen_params.get('do_sample', True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
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
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Извлечение JSON из текста ответа модели"""
        # Ищем JSON в тексте
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                # Исправляем common issues
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга JSON: {e}. Текст: {json_str[:200]}")
        
        # Попытка найти JSON-массив
        array_match = re.search(r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', text, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                json_str = json_str.replace("'", '"')
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def detect_scenes(self, screenplay_text: str) -> List[SceneDetection]:
        """Шаг 1: Детекция сцен через LLM"""
        logger.info("Начинаю детекцию сцен через LLM...")
        
        # Разбиваем текст на части для обработки (если слишком большой)
        max_chunk_size = 20000
        if len(screenplay_text) > max_chunk_size:
            # Обрабатываем по частям
            chunks = []
            for i in range(0, len(screenplay_text), max_chunk_size):
                chunks.append(screenplay_text[i:i+max_chunk_size])
        else:
            chunks = [screenplay_text]
        
        all_scenes = []
        
        for chunk in chunks:
            prompt = self._generate_scene_detection_prompt(chunk)
            response = self._call_model([prompt])[0]
            
            # Извлекаем JSON
            result = self._extract_json(response)
            
            if result and isinstance(result, list):
                for scene_data in result:
                    scene = SceneDetection(
                        scene_number=str(scene_data.get('scene_number', '')),
                        scene_title=scene_data.get('scene_title', ''),
                        scene_text='',  # Будет заполнено позже
                        start_pos=scene_data.get('start_pos', 0),
                        end_pos=scene_data.get('end_pos', 0),
                        confidence=0.9
                    )
                    all_scenes.append(scene)
            else:
                logger.warning("Не удалось извлечь сцены из ответа LLM, используем fallback")
                # Fallback: разбиение по заголовкам
                all_scenes.extend(self._fallback_scene_detection(chunk))
        
        # Извлекаем текст сцен из оригинального текста
        for scene in all_scenes:
            if scene.start_pos < len(screenplay_text) and scene.end_pos <= len(screenplay_text):
                scene.scene_text = screenplay_text[scene.start_pos:scene.end_pos]
            else:
                # Если позиции некорректны, используем альтернативный метод
                scene.scene_text = self._extract_scene_text_by_title(screenplay_text, scene.scene_title)
        
        logger.info(f"Обнаружено {len(all_scenes)} сцен")
        return all_scenes
    
    def _fallback_scene_detection(self, text: str) -> List[SceneDetection]:
        """Fallback метод детекции сцен через regex"""
        scenes = []
        pattern = re.compile(
            r'(?P<number>\d+[-.]?\d*\.? )?\s*'
            r'(?P<type>INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.)\s*'
            r'(?P<location>[^.\n]+?)(?:\.\s*(?P<sublocation>[^.\n]+?))?\s*[.\-\s]*\s*'
            r'(?P<time>ДЕНЬ|НОЧЬ|УТРО|ВЕЧЕР|РАССВЕТ|ЗАКАТ)?',
            re.MULTILINE | re.IGNORECASE
        )
        
        matches = list(pattern.finditer(text))
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            scene = SceneDetection(
                scene_number=str(match.group('number') or i + 1),
                scene_title=match.group(0),
                scene_text=text[start:end],
                start_pos=start,
                end_pos=end,
                confidence=0.7
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
                prompt = self._generate_entity_extraction_prompt(
                    scene.scene_text,
                    scene.scene_number,
                    entities_to_extract
                )
                prompts.append(prompt)
            
            # Вызываем модель батчем
            responses = self._call_model(prompts)
            
            # Парсим результаты
            for scene, response in zip(batch, responses):
                entities = self._extract_json(response) or {}
                
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
            result = {
                'scene_number': scene.scene_number,
                'scene_title': scene.scene_title,
                'scene_text': scene.scene_text,
                **extraction.entities
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

