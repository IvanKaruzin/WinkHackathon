Создам полноценный пайплайн с использованием локальной LLM, оптимизированной для MacBook M3 Pro. Буду использовать Llama.cpp с квантованными моделями для экономии памяти.

## Полный код решения

### 1. Основной файл `screenplay_parser.py`:

```python
#!/usr/bin/env python3
"""
screenplay_parser.py

Сервис парсинга типизированных сценариев в excel-таблицу.
Использует локальную LLM через llama-cpp-python для извлечения метаданных.

Автор: Production Pipeline Parser
Версия: 1.0.0
"""

import argparse
import json
import os
import re
import sys
import gc
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from docx import Document
from tqdm import tqdm
import numpy as np

# Импорт llama-cpp
try:
    from llama_cpp import Llama
except ImportError:
    print("Ошибка: llama-cpp-python не установлен")
    print("Установите: pip install llama-cpp-python")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('screenplay_parser.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
#  Конфигурация
# -----------------------------

class Config:
    """Конфигурация парсера"""
    # Пути
    MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # Путь к модели
    
    # Параметры модели для M3 Pro
    MODEL_PARAMS = {
        'n_ctx': 2048,        # Контекст (уменьшен для экономии памяти)
        'n_batch': 512,       # Размер батча
        'n_threads': 8,       # Количество потоков (для M3 Pro)
        'n_gpu_layers': 1,    # Использование Metal (GPU на Mac)
        'use_mmap': True,     # Использование memory mapping
        'use_mlock': False,   # Не блокировать память
        'seed': 42,
        'verbose': False
    }
    
    # Параметры генерации
    GENERATION_PARAMS = {
        'max_tokens': 512,
        'temperature': 0.3,
        'top_p': 0.95,
        'top_k': 40,
        'repeat_penalty': 1.1,
        'stop': ["</s>", "\n\n\n", "---"]
    }
    
    # Парсинг
    MIN_SCENE_LENGTH = 50
    MAX_SCENE_LENGTH = 5000
    BATCH_SIZE = 5  # Обрабатывать по 5 сцен за раз

# -----------------------------
#  Структуры данных
# -----------------------------

@dataclass
class SceneMetadata:
    """Структура для хранения метаданных сцены"""
    scene_number: str = ""
    episode: str = ""
    scene_type: str = ""  # INT/EXT/ИНТ/ЭКСТ/НАТ
    location: str = ""
    sublocation: str = ""
    time_of_day: str = ""
    synopsis: str = ""
    characters: List[str] = field(default_factory=list)
    extras: str = ""
    extras_count: int = 0
    props: List[str] = field(default_factory=list)
    vehicles: List[str] = field(default_factory=list)
    special_fx: List[str] = field(default_factory=list)
    costumes: List[str] = field(default_factory=list)
    makeup: List[str] = field(default_factory=list)
    stunts: bool = False
    pyrotechnics: bool = False
    special_equipment: List[str] = field(default_factory=list)
    notes: str = ""
    raw_text: str = ""
    confidence_score: float = 0.0

# -----------------------------
#  LLM Manager
# -----------------------------

class LocalLLM:
    """Менеджер для работы с локальной LLM"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель с оптимизацией под Mac M3"""
        try:
            logger.info(f"Загрузка модели из {self.model_path}")
            logger.info("Это может занять 1-2 минуты...")
            
            # Проверяем существование файла модели
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
            
            # Инициализация с параметрами для M3 Pro
            self.model = Llama(
                model_path=self.model_path,
                **Config.MODEL_PARAMS
            )
            
            logger.info("Модель успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.info("Работаем без LLM, используя только правила")
            self.model = None
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Генерирует ответ на промпт"""
        if self.model is None:
            return "{}"
        
        try:
            # Формируем полный промпт
            if system_prompt:
                full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
            else:
                full_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Генерация
            response = self.model(
                full_prompt,
                **Config.GENERATION_PARAMS
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.warning(f"Ошибка генерации: {e}")
            return "{}"
    
    def extract_json(self, text: str) -> Dict[str, Any]:
        """Извлекает JSON из ответа модели"""
        try:
            # Ищем JSON в тексте
            json_match = re.search(r'\{[^}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Исправляем common issues
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                return json.loads(json_str)
        except:
            pass
        return {}
    
    def __del__(self):
        """Освобождение памяти при удалении объекта"""
        if self.model:
            del self.model
            gc.collect()

# -----------------------------
#  Парсинг сценария
# -----------------------------

class ScenarioParser:
    """Класс для парсинга сценария"""
    
    # Паттерны для определения элементов
    SCENE_PATTERNS = {
        'heading': re.compile(
            r'^(?P<number>\d+[-.]?\d*\.?)?\s*'
            r'(?P<type>INT\.|EXT\.|ИНТ\.|ЭКСТ\.|НАТ\.)\s*'
            r'(?P<location>[^.\n]+?)(?:\.\s*(?P<sublocation>[^.\n]+?))?\s*[\.\-\s]*\s*'
            r'(?P<time>ДЕНЬ|НОЧЬ|УТРО|ВЕЧЕР|РАССВЕТ|ЗАКАТ|День|Ночь|Утро|Вечер)?',
            re.MULTILINE | re.IGNORECASE
        ),
        'character': re.compile(
            r'^([А-ЯЁA-Z][А-ЯЁA-Z\s\-,]{1,30})(?:\s*\([\w\s,]+\))?$',
            re.MULTILINE
        ),
        'parenthetical': re.compile(
            r'\(([^)]+)\)',
            re.MULTILINE
        )
    }
    
    # Ключевые слова для извлечения
    KEYWORDS = {
        'props': [
            'телефон', 'ноутбук', 'компьютер', 'письмо', 'книга', 'сумка',
            'ключи', 'документы', 'оружие', 'нож', 'пистолет', 'деньги',
            'фотография', 'камера', 'микрофон', 'наушники', 'очки', 'часы',
            'кольцо', 'цветы', 'бутылка', 'стакан', 'еда', 'напиток'
        ],
        'vehicles': [
            'машина', 'автомобиль', 'автобус', 'такси', 'мотоцикл',
            'велосипед', 'самолет', 'вертолет', 'поезд', 'корабль', 'лодка'
        ],
        'effects': [
            'взрыв', 'выстрел', 'дым', 'огонь', 'пожар', 'искры', 'кровь',
            'слезы', 'дождь', 'снег', 'туман', 'ветер', 'молния', 'гром'
        ],
        'stunts': [
            'драка', 'удар', 'падение', 'прыжок', 'погоня', 'авария',
            'бег', 'борьба', 'трюк', 'каскадер'
        ]
    }
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = None
        if use_llm:
            self.llm = LocalLLM()
        self.scenes = []
        
    def parse_screenplay(self, text: str) -> List[SceneMetadata]:
        """Основной метод парсинга сценария"""
        logger.info("Начинаю парсинг сценария...")
        
        # Разбиваем на сцены
        scenes_raw = self._split_into_scenes(text)
        logger.info(f"Найдено {len(scenes_raw)} сцен")
        
        # Обрабатываем сцены батчами
        batch_size = Config.BATCH_SIZE
        for i in tqdm(range(0, len(scenes_raw), batch_size), desc="Обработка сцен"):
            batch = scenes_raw[i:i + batch_size]
            
            for j, scene_text in enumerate(batch):
                scene_num = i + j + 1
                metadata = self._extract_scene_metadata(scene_text, scene_num)
                
                # Улучшаем с помощью LLM если доступна
                if self.use_llm and self.llm and self.llm.model:
                    metadata = self._enhance_with_llm(metadata)
                
                self.scenes.append(metadata)
            
            # Очистка памяти после батча
            gc.collect()
        
        return self.scenes
    
    def _split_into_scenes(self, text: str) -> List[str]:
        """Разбивает текст на отдельные сцены"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Ищем заголовки сцен
        scenes = []
        scene_headers = list(self.SCENE_PATTERNS['heading'].finditer(text))
        
        if not scene_headers:
            logger.warning("Явные заголовки сцен не найдены, используем альтернативное разбиение")
            # Альтернативное разбиение по двойным переводам строки
            parts = re.split(r'\n{2,}', text)
            return [p.strip() for p in parts 
                   if p and Config.MIN_SCENE_LENGTH <= len(p.strip()) <= Config.MAX_SCENE_LENGTH]
        
        for i, match in enumerate(scene_headers):
            start = match.start()
            end = scene_headers[i + 1].start() if i + 1 < len(scene_headers) else len(text)
            scene_text = text[start:end].strip()
            
            if Config.MIN_SCENE_LENGTH <= len(scene_text) <= Config.MAX_SCENE_LENGTH:
                scenes.append(scene_text)
        
        return scenes
    
    def _extract_scene_metadata(self, scene_text: str, scene_num: int) -> SceneMetadata:
        """Извлекает базовые метаданные из текста сцены"""
        metadata = SceneMetadata(
            scene_number=str(scene_num),
            raw_text=scene_text[:500]
        )
        
        # Заголовок сцены
        header_match = self.SCENE_PATTERNS['heading'].search(scene_text)
        if header_match:
            groups = header_match.groupdict()
            metadata.scene_number = groups.get('number') or str(scene_num)
            metadata.scene_type = (groups.get('type') or 'INT').strip('.')
            metadata.location = (groups.get('location') or '').strip()
            metadata.sublocation = (groups.get('sublocation') or '').strip()
            metadata.time_of_day = groups.get('time') or 'ДЕНЬ'
        
        # Извлекаем персонажей
        metadata.characters = self._extract_characters(scene_text)
        
        # Извлекаем синопсис
        metadata.synopsis = self._extract_synopsis(scene_text)
        
        # Извлекаем элементы по ключевым словам
        text_lower = scene_text.lower()
        
        # Реквизит
        metadata.props = [prop for prop in self.KEYWORDS['props'] 
                         if prop in text_lower][:10]
        
        # Транспорт
        metadata.vehicles = [v for v in self.KEYWORDS['vehicles'] 
                           if v in text_lower][:5]
        
        # Спецэффекты
        metadata.special_fx = [fx for fx in self.KEYWORDS['effects'] 
                              if fx in text_lower]
        
        # Трюки
        metadata.stunts = any(stunt in text_lower for stunt in self.KEYWORDS['stunts'])
        
        # Пиротехника
        metadata.pyrotechnics = any(word in text_lower for word in ['взрыв', 'огонь', 'пожар', 'выстрел'])
        
        # Массовка (простая эвристика)
        extras_match = re.search(r'(?:массовка|толпа|зрители|прохожие|студенты|гости)[\s:\-]*(\d+)?', 
                                 text_lower)
        if extras_match:
            metadata.extras = extras_match.group(0)
            if extras_match.group(1):
                metadata.extras_count = int(extras_match.group(1))
        
        return metadata
    
    def _extract_characters(self, text: str) -> List[str]:
        """Извлекает имена персонажей"""
        characters = set()
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Проверяем паттерн для имени персонажа
            if self.SCENE_PATTERNS['character'].match(line):
                # Убираем ремарки в скобках
                character = re.sub(r'\([^)]*\)', '', line).strip()
                
                # Фильтруем заголовки сцен и технические термины
                if (character and 
                    not any(word in character.upper() for word in 
                           ['ИНТ', 'ЭКСТ', 'НАТ', 'ДЕНЬ', 'НОЧЬ', 'УТРО', 'ВЕЧЕР']) and
                    len(character) > 2):
                    characters.add(character)
        
        # Дополнительный поиск в тексте
        # Ищем имена с большой буквы после определенных слов
        name_contexts = re.findall(
            r'(?:говорит|спрашивает|отвечает|кричит|шепчет|зовет)\s+([А-ЯЁ][а-яё]+)',
            text
        )
        characters.update(name_contexts)
        
        return sorted(list(characters))[:15]  # Ограничиваем 15 персонажами
    
    def _extract_synopsis(self, text: str) -> str:
        """Извлекает краткое описание действия"""
        lines = text.split('\n')
        synopsis_lines = []
        
        # Пропускаем заголовок
        start_idx = 0
        for i, line in enumerate(lines):
            if self.SCENE_PATTERNS['heading'].match(line):
                start_idx = i + 1
                break
        
        # Собираем описательные строки
        for line in lines[start_idx:]:
            line = line.strip()
            
            # Останавливаемся на диалогах
            if self.SCENE_PATTERNS['character'].match(line):
                break
                
            # Добавляем описательные строки
            if line and not line.isupper():
                synopsis_lines.append(line)
                if len(' '.join(synopsis_lines)) > 300:
                    break
        
        synopsis = ' '.join(synopsis_lines)
        # Очищаем от лишних пробелов
        synopsis = ' '.join(synopsis.split())
        
        return synopsis[:400]
    
    def _enhance_with_llm(self, metadata: SceneMetadata) -> SceneMetadata:
        """Улучшает метаданные с помощью LLM"""
        if not self.llm or not self.llm.model:
            return metadata
        
        try:
            system_prompt = """Ты - ассистент режиссера, анализирующий сценарии для кинопроизводства.
Твоя задача - извлечь точную информацию о производственных требованиях сцены.
Отвечай ТОЛЬКО в формате JSON, без дополнительного текста."""

            prompt = f"""Проанализируй сцену и извлеки недостающую информацию:

СЦЕНА: {metadata.location} - {metadata.time_of_day}
ТЕКСТ: {metadata.raw_text}

Уже извлечено:
- Персонажи: {', '.join(metadata.characters[:5]) if metadata.characters else 'не найдены'}
- Реквизит: {', '.join(metadata.props[:5]) if metadata.props else 'не найден'}

Дополни в формате JSON:
{{
  "extras_description": "описание массовки и количество",
  "additional_props": ["дополнительный", "реквизит"],
  "costume_notes": ["особенности", "костюмов"],
  "makeup_notes": ["требования", "к гриму"],
  "special_requirements": "особые требования к съемке"
}}"""

            # Генерируем ответ
            response = self.llm.generate(prompt, system_prompt)
            
            # Парсим JSON
            data = self.llm.extract_json(response)
            
            if data:
                # Обновляем метаданные
                if 'extras_description' in data:
                    metadata.extras = str(data['extras_description'])
                    
                if 'additional_props' in data and isinstance(data['additional_props'], list):
                    metadata.props.extend(data['additional_props'])
                    metadata.props = list(set(metadata.props))[:15]
                    
                if 'costume_notes' in data and isinstance(data['costume_notes'], list):
                    metadata.costumes = data['costume_notes'][:5]
                    
                if 'makeup_notes' in data and isinstance(data['makeup_notes'], list):
                    metadata.makeup = data['makeup_notes'][:5]
                    
                if 'special_requirements' in data:
                    metadata.notes = str(data['special_requirements'])
                
                metadata.confidence_score = 0.8  # LLM-enhanced
            else:
                metadata.confidence_score = 0.5  # Rules only
                
        except Exception as e:
            logger.warning(f"Ошибка при улучшении с LLM: {e}")
            metadata.confidence_score = 0.5
        
        return metadata

# -----------------------------
#  Чтение файлов
# -----------------------------

def read_docx(path: str) -> str:
    """Читает .docx файл"""
    try:
        doc = Document(path)
        paragraphs = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Сохраняем форматирование
                if para.style.name.startswith('Heading'):
                    text = f"\n\n{text}\n"
                paragraphs.append(text)
        
        return "\n".join(paragraphs)
        
    except Exception as e:
        logger.error(f"Ошибка чтения файла {path}: {e}")
        raise

# -----------------------------
#  Экспорт в Excel
# -----------------------------

def create_production_table(scenes: List[SceneMetadata]) -> pd.DataFrame:
    """Создает таблицу для pre-production"""
    rows = []
    
    for scene in scenes:
        row = {
            "Серия": scene.episode or "01",
            "Сцена": scene.scene_number,
            "Режим": scene.time_of_day,
            "Инт/Нат": scene.scene_type,
            "Объект": scene.location,
            "Подобъект": scene.sublocation,
            "Синопсис": scene.synopsis[:200] if scene.synopsis else "",
            "Персонажи": ", ".join(scene.characters[:8]) if scene.characters else "",
            "Массовка": scene.extras,
            "Кол-во массовки": scene.extras_count if scene.extras_count else "",
            "Реквизит": ", ".join(scene.props[:8]) if scene.props else "",
            "Игровой транспорт": ", ".join(scene.vehicles) if scene.vehicles else "",
            "Художники": "",  # Заполняется вручную
            "Грим": ", ".join(scene.makeup) if scene.makeup else "",
            "Костюм": ", ".join(scene.costumes) if scene.costumes else "",
            "Каскадеры": "Да" if scene.stunts else "",
            "Пиротехника": "Да" if scene.pyrotechnics else "",
            "Спец. оборудование": ", ".join(scene.special_equipment) if scene.special_equipment else "",
            "Примечание": scene.notes,
            "Уверенность": f"{scene.confidence_score:.0%}" if scene.confidence_score > 0 else ""
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Сортировка по номеру сцены
    try:
        df['scene_num'] = df['Сцена'].str.extract(r'(\d+)').astype(float)
        df = df.sort_values('scene_num').drop('scene_num', axis=1)
    except:
        pass
    
    return df

def export_to_excel(df: pd.DataFrame, output_path: str):
    """Экспортирует DataFrame в Excel с форматированием"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        # Создаем workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "КПП"
        
        # Добавляем заголовок
        ws.append([f"КПП - Календарно-постановочный план"])
        ws.merge_cells('A1:T1')
        
        # Стиль заголовка
        header_font = Font(bold=True, size=14)
        ws['A1'].font = header_font
        ws['A1'].alignment = Alignment(horizontal='center')
        
        # Пустая строка
        ws.append([])
        
        # Добавляем данные
        for r in dataframe_to_rows(df, index=False, header=True):
            ws.append(r)
        
        # Форматирование заголовков столбцов
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        for cell in ws[3]:  # Третья строка - заголовки столбцов
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        
        # Настройка ширины столбцов
        column_widths = {
            'A': 8,   # Серия
            'B': 10,  # Сцена
            'C': 10,  # Режим
            'D': 12,  # Инт/Нат
            'E': 25,  # Объект
            'F': 25,  # Подобъект
            'G': 40,  # Синопсис
            'H': 30,  # Персонажи
            'I': 20,  # Массовка
            'J': 10,  # Кол-во
            'K': 30,  # Реквизит
            'L': 20,  # Транспорт
            'M': 20,  # Художники
            'N': 20,  # Грим
            'O': 20,  # Костюм
            'P': 12,  # Каскадеры
            'Q': 12,  # Пиротехника
            'R': 25,  # Спец. оборудование
            'S': 30,  # Примечание
            'T': 12   # Уверенность
        }
        
        for col, width in column_widths.items():
            ws.column_dimensions[col].width = width
        
        # Границы для всех ячеек с данными
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in ws.iter_rows(min_row=3, max_row=ws.max_row):
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        
        # Условное форматирование для уверенности
        for row in ws.iter_rows(min_row=4, max_row=ws.max_row, min_col=20, max_col=20):
            for cell in row:
                if cell.value:
                    try:
                        confidence = float(cell.value.strip('%')) / 100
                        if confidence >= 0.8:
                            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                        elif confidence >= 0.6:
                            cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                        else:
                            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                    except:
                        pass
        
        # Сохраняем
        wb.save(output_path)
        logger.info(f"Таблица успешно экспортирована в {output_path}")
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте в Excel: {e}")
        # Fallback на простой экспорт pandas
        df.to_excel(output_path, index=False, sheet_name='КПП')
        logger.info(f"Таблица экспортирована в упрощенном формате")

# -----------------------------
#  Статистика и отчеты
# -----------------------------

def print_statistics(scenes: List[SceneMetadata], df: pd.DataFrame):
    """Выводит статистику по сценарию"""
    print("\n" + "="*70)
    print(" " * 20 + "СТАТИСТИКА ПАРСИНГА")
    print("="*70)
    
    # Основная статистика
    print(f"\n📊 ОСНОВНЫЕ ПОКАЗАТЕЛИ:")
    print(f"  • Всего сцен: {len(scenes)}")
    print(f"  • Интерьеры: {sum(1 for s in scenes if s.scene_type in ['INT', 'ИНТ'])}")
    print(f"  • Натура: {sum(1 for s in scenes if s.scene_type in ['EXT', 'ЭКСТ', 'НАТ'])}")
    print(f"  • Дневные сцены: {sum(1 for s in scenes if 'ДЕНЬ' in s.time_of_day.upper())}")
    print(f"  • Ночные сцены: {sum(1 for s in scenes if 'НОЧЬ' in s.time_of_day.upper())}")
    
    # Локации
    locations = df['Объект'].value_counts()
    print(f"\n📍 ЛОКАЦИИ:")
    print(f"  • Уникальных локаций: {len(locations)}")
    print(f"  • Топ-5 локаций:")
    for loc, count in locations.head(5).items():
        print(f"    - {loc}: {count} сцен")
    
    # Персонажи
    all_characters = []
    for scene in scenes:
        all_characters.extend(scene.characters)
    unique_chars = list(set(all_characters))
    
    print(f"\n👥 ПЕРСОНАЖИ:")
    print(f"  • Уникальных персонажей: {len(unique_chars)}")
    if unique_chars:
        from collections import Counter
        char_counts = Counter(all_characters)
        print(f"  • Топ-5 персонажей:")
        for char, count in char_counts.most_common(5):
            print(f"    - {char}: {count} сцен")
    
    # Производственные требования
    print(f"\n🎬 ПРОИЗВОДСТВЕННЫЕ ТРЕБОВАНИЯ:")
    print(f"  • Сцены с массовкой: {sum(1 for s in scenes if s.extras)}")
    print(f"  • Сцены с транспортом: {sum(1 for s in scenes if s.vehicles)}")
    print(f"  • Сцены со спецэффектами: {sum(1 for s in scenes if s.special_fx)}")
    print(f"  • Сцены с трюками: {sum(1 for s in scenes if s.stunts)}")
    print(f"  • Сцены с пиротехникой: {sum(1 for s in scenes if s.pyrotechnics)}")
    
    # Качество парсинга
    if any(s.confidence_score > 0 for s in scenes):
        avg_confidence = np.mean([s.confidence_score for s in scenes if s.confidence_score > 0])
        print(f"\n📈 КАЧЕСТВО ПАРСИНГА:")
        print(f"  • Средняя уверенность: {avg_confidence:.0%}")
        print(f"  • Сцены с высокой уверенностью (>80%): {sum(1 for s in scenes if s.confidence_score > 0.8)}")
        print(f"  • Сцены с низкой уверенностью (<60%): {sum(1 for s in scenes if 0 < s.confidence_score < 0.6)}")
    
    print("\n" + "="*70)

# -----------------------------
#  Основная функция
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🎬 Парсер сценариев для создания КПП (календарно-постановочного плана)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python screenplay_parser.py -i scenario.docx -o production.xlsx
  python screenplay_parser.py -i scenario.docx --no-llm  # без LLM
  python screenplay_parser.py -i scenario.docx --model models/my_model.gguf
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Путь к файлу сценария (.docx)"
    )
    parser.add_argument(
        "--output", "-o",
        default="production_table.xlsx",
        help="Путь для сохранения Excel таблицы (по умолчанию: production_table.xlsx)"
    )
    parser.add_argument(
        "--model",
        default=Config.MODEL_PATH,
        help="Путь к файлу модели GGUF"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Не использовать LLM (только правила)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Размер батча для обработки сцен"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Режим отладки с подробным выводом"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Проверки
    if not os.path.exists(args.input):
        logger.error(f"❌ Файл не найден: {args.input}")
        sys.exit(1)
    
    if not args.input.endswith('.docx'):
        logger.error("❌ Поддерживается только формат .docx")
        sys.exit(1)
    
    # Обновляем конфигурацию
    Config.MODEL_PATH = args.model
    Config.BATCH_SIZE = args.batch_size
    
    try:
        # Читаем файл
        logger.info(f"📖 Чтение файла {args.input}...")
        text = read_docx(args.input)
        logger.info(f"✓ Прочитано {len(text)} символов")
        
        # Создаем парсер
        use_llm = not args.no_llm
        if use_llm:
            logger.info("🤖 Инициализация LLM...")
        else:
            logger.info("📝 Работа в режиме без LLM (только правила)")
        
        parser_obj = ScenarioParser(use_llm=use_llm)
        
        # Парсим сценарий
        scenes = parser_obj.parse_screenplay(text)
        logger.info(f"✓ Обработано сцен: {len(scenes)}")
        
        # Создаем таблицу
        logger.info("📊 Создание таблицы production...")
        df = create_production_table(scenes)
        
        # Экспортируем
        logger.info(f"💾 Сохранение в {args.output}...")
        export_to_excel(df, args.output)
        
        # Выводим статистику
        print_statistics(scenes, df)
        
        print(f"\n✅ ГОТОВО! Файл сохранен: {args.output}")
        print(f"📂 Откройте файл в Excel для просмотра и редактирования")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Очистка памяти
        gc.collect()

if __name__ == "__main__":
    main()
```

### 2. Файл установки зависимостей `requirements.txt`:

```txt
# Основные библиотеки
pandas>=2.0.0
openpyxl>=3.1.0
python-docx>=0.8.11
tqdm>=4.65.0
numpy>=1.24.0

# LLM
llama-cpp-python>=0.2.20

# Опциональные для улучшения
XlsxWriter>=3.1.0
```

### 3. Скрипт установки модели `setup_model.sh`:

```bash
#!/bin/bash
# setup_model.sh - Скрипт для загрузки и установки модели

echo "🚀 Настройка окружения для парсера сценариев"
echo "============================================"

# Создание директорий
echo "📁 Создание структуры папок..."
mkdir -p models
mkdir -p output
mkdir -p logs
mkdir -p input

# Загрузка модели
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Загрузка модели Mistral 7B (это займет время, размер ~4GB)..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
    echo "✅ Модель загружена!"
else
    echo "✅ Модель уже существует"
fi

echo ""
echo "✨ Установка завершена!"
echo "Теперь вы можете использовать парсер:"
echo "python screenplay_parser.py -i input/scenario.docx -o output/production.xlsx"
```

## 📋 ПОЛНЫЙ ПАЙПЛАЙН УСТАНОВКИ И ИСПОЛЬЗОВАНИЯ

### Шаг 1: Подготовка системы

```bash
# 1.1 Установка Homebrew (если еще нет)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 1.2 Установка Python 3.11 (рекомендуется)
brew install python@3.11

# 1.3 Установка cmake (нужно для llama-cpp)
brew install cmake
```

### Шаг 2: Создание проекта

```bash
# 2.1 Создание папки проекта
mkdir screenplay_parser_project
cd screenplay_parser_project

# 2.2 Создание виртуального окружения
python3.11 -m venv venv

# 2.3 Активация окружения
source venv/bin/activate
```

### Шаг 3: Установка зависимостей

```bash
# 3.1 Обновление pip
pip install --upgrade pip

# 3.2 Установка llama-cpp-python с поддержкой Metal (GPU на Mac)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# 3.3 Установка остальных зависимостей
pip install pandas openpyxl python-docx tqdm numpy XlsxWriter
```

### Шаг 4: Настройка файлов

```bash
# 4.1 Создайте файл screenplay_parser.py и скопируйте код выше

# 4.2 Создайте структуру папок
mkdir -p models output logs input

# 4.3 Сделайте скрипт установки исполняемым
chmod +x setup_model.sh
```

### Шаг 5: Загрузка модели

```bash
# Вариант 1: Автоматическая загрузка
./setup_model.sh

# Вариант 2: Ручная загрузка более легкой модели (2.5GB)
curl -L -o models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Шаг 6: Запуск парсера

```bash
# 6.1 Базовый запуск с LLM
python screenplay_parser.py -i input/scenario.docx -o output/production.xlsx

# 6.2 Без LLM (быстрее, но менее точно)
python screenplay_parser.py -i input/scenario.docx -o output/production.xlsx --no-llm

# 6.3 С отладкой
python screenplay_parser.py -i input/scenario.docx -o output/production.xlsx --debug

# 6.4 С маленькими батчами (меньше памяти)
python screenplay_parser.py -i input/scenario.docx --batch-size 3
```

## 🎯 ОПТИМИЗАЦИЯ ДЛЯ MAC M3 PRO

### Рекомендуемые модели по размеру:

1. **Минимальная (2GB)** - для 8GB RAM:
```bash
curl -L -o models/model.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

2. **Оптимальная (4GB)** - для 16GB RAM:
```bash
curl -L -o models/model.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

3. **Максимальная (5.5GB)** - для 32GB+ RAM:
```bash
curl -L -o models/model.gguf \
  https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf
```

### Мониторинг производительности:

```bash
# Проверка использования памяти
top -l 1 | grep PhysMem

# Проверка активности GPU
sudo powermetrics --samplers gpu_power -i1000 -n1
```

## 📊 ПРОВЕРКА РЕЗУЛЬТАТОВ

После запуска вы получите:

1. **Excel файл** с таблицей КПП
2. **Лог файл** `screenplay_parser.log`
3. **Консольный вывод** со статистикой

### Структура выходной таблицы:
- Серия/Эпизод
- Номер сцены
- Время суток
- Интерьер/Натура
- Локация и подлокация
- Синопсис действия
- Список персонажей
- Массовка с количеством
- Реквизит
- Транспорт
- Требования к гриму/костюмам
- Спецэффекты и трюки
- Примечания
- Уровень уверенности парсинга

## 🔧 РЕШЕНИЕ ПРОБЛЕМ

### Если модель не загружается:
```bash
# Используйте режим без LLM
python screenplay_parser.py -i input.docx --no-llm
```

### Если не хватает памяти:
```bash
# Уменьшите размер батча
python screenplay_parser.py -i input.docx --batch-size 1
```

### Если Metal не работает:
```bash
# Переустановите без Metal
pip uninstall llama-cpp-python
pip install llama-cpp-python
```

## ✅ ГОТОВО!

Теперь у вас есть полностью рабочий парсер сценариев с локальной LLM, оптимизированный для MacBook M3 Pro!