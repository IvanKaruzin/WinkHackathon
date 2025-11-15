#!/usr/bin/env python3
"""
result_validator.py

Валидация и постобработка результатов извлечения сущностей для исправления типичных ошибок LLM.
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ResultValidator:
    """Валидатор и очиститель результатов извлечения сущностей"""
    
    # Словари для валидации
    INVALID_LOCATION_PATTERNS = [
        r'отделаются',
        r'плавают',
        r'идут',
        r'стоят',
        r'сидят',
        r'бегут',
        r'говорят',
        r'смотрят',
        r'делают',
        r'находятся',
    ]
    
    VALID_EQUIPMENT_KEYWORDS = [
        'кран', 'дрон', 'стабилизатор', 'тележка', 'журавль', 'микрофон',
        'освещение', 'камера', 'оператор', 'съемочн', 'техническ',
        'подвес', 'трос', 'подъемник', 'платформа', 'рельс'
    ]
    
    INVALID_EQUIPMENT_PATTERNS = [
        r'стол', r'стул', r'кровать', r'диван', r'телефон', r'компьютер',
        r'книга', r'бумага', r'ручка', r'одежда', r'костюм', r'грим',
        r'макияж', r'реквизит'
    ]
    
    def __init__(self):
        self.invalid_location_re = re.compile(
            '|'.join(self.INVALID_LOCATION_PATTERNS),
            re.IGNORECASE
        )
        self.invalid_equipment_re = re.compile(
            '|'.join(self.INVALID_EQUIPMENT_PATTERNS),
            re.IGNORECASE
        )
    
    def validate_and_clean(self, entities: Dict[str, Any], scene_text: str) -> Dict[str, Any]:
        """
        Валидирует и очищает извлеченные сущности
        
        Args:
            entities: Словарь с извлеченными сущностями
            scene_text: Оригинальный текст сцены для контекста
            
        Returns:
            Очищенный словарь сущностей
        """
        cleaned = entities.copy()
        
        # Валидация локации
        cleaned['location'] = self._validate_location(
            cleaned.get('location', ''),
            scene_text
        )
        
        # Валидация персонажей
        cleaned['characters'] = self._validate_characters(
            cleaned.get('characters', []),
            scene_text
        )
        
        # Валидация спецоборудования
        cleaned['special_equipment'] = self._validate_equipment(
            cleaned.get('special_equipment', []),
            scene_text
        )
        
        # Валидация количества массовки
        cleaned['crowd_count'] = self._validate_crowd_count(
            cleaned.get('crowd_count', 0),
            cleaned.get('crowd', ''),
            scene_text
        )
        
        # Валидация костюмов
        cleaned['costumes'] = self._validate_costumes(
            cleaned.get('costumes', []),
            scene_text
        )
        
        return cleaned
    
    def _validate_location(self, location: str, scene_text: str) -> str:
        """Валидация локации - убираем глаголы и действия"""
        if not location:
            return location
        
        location_lower = location.lower()
        
        # Проверяем, не является ли локация глаголом или действием
        if self.invalid_location_re.search(location):
            logger.warning(f"Обнаружена некорректная локация: '{location}'. Пытаюсь исправить...")
            
            # Пытаемся найти настоящую локацию в тексте сцены
            # Ищем паттерны типа "ИНТ. ЛОКАЦИЯ" или "ЭКСТ. ЛОКАЦИЯ"
            location_pattern = re.compile(
                r'(?:ИНТ|ЭКСТ|НАТ|INT|EXT)\.\s*([А-ЯЁ\w\s]+?)(?:\s*[-–—]\s*|\.|$)',
                re.IGNORECASE
            )
            match = location_pattern.search(scene_text)
            if match:
                found_location = match.group(1).strip()
                if found_location and not self.invalid_location_re.search(found_location):
                    logger.info(f"Исправлена локация: '{location}' -> '{found_location}'")
                    return found_location
            
            # Если не нашли, возвращаем пустую строку
            return ""
        
        return location
    
    def _validate_characters(self, characters: List[str], scene_text: str) -> List[str]:
        """Валидация персонажей - убираем массовку и действия"""
        if not characters:
            return characters
        
        valid_characters = []
        
        for char in characters:
            char_str = str(char).strip()
            if not char_str:
                continue
            
            char_lower = char_str.lower()
            
            # Пропускаем слова, которые не являются именами
            if any(word in char_lower for word in ['толпа', 'массовка', 'люди', 'прохожие', 'студенты', 'официанты']):
                continue
            
            # Пропускаем глаголы и действия
            if self.invalid_location_re.search(char_str):
                continue
            
            # Пропускаем слишком длинные "имена" (вероятно, это описание)
            if len(char_str) > 30:
                continue
            
            valid_characters.append(char_str)
        
        return valid_characters
    
    def _validate_equipment(self, equipment: List[str], scene_text: str) -> List[str]:
        """Валидация спецоборудования - только профессиональное съемочное оборудование"""
        if not equipment:
            return equipment
        
        valid_equipment = []
        
        for item in equipment:
            item_str = str(item).strip().lower()
            if not item_str:
                continue
            
            # Проверяем, что это не обычный реквизит
            if self.invalid_equipment_re.search(item_str):
                logger.warning(f"Удалено из спецоборудования (это реквизит): '{item}'")
                continue
            
            # Проверяем, что это профессиональное оборудование
            if any(keyword in item_str for keyword in self.VALID_EQUIPMENT_KEYWORDS):
                valid_equipment.append(item)
            else:
                # Если не похоже на профессиональное оборудование, пропускаем
                logger.warning(f"Удалено из спецоборудования (не профессиональное): '{item}'")
        
        return valid_equipment
    
    def _validate_crowd_count(self, crowd_count: Any, crowd: str, scene_text: str) -> int:
        """Валидация количества массовки"""
        # Если crowd_count уже есть и валидный
        if isinstance(crowd_count, (int, float)) and crowd_count > 0:
            return int(crowd_count)
        
        # Пытаемся извлечь из текста crowd
        if crowd:
            numbers = re.findall(r'\d+', str(crowd))
            if numbers:
                try:
                    return int(numbers[0])
                except (ValueError, TypeError):
                    pass
        
        # Пытаемся извлечь из текста сцены
        crowd_patterns = [
            r'массовк[аи]\s*[:—]\s*(\d+)',
            r'толп[аеы]\s*из\s*(\d+)',
            r'(\d+)\s*человек',
            r'(\d+)\s*чел\.',
            r'(\d+)\s*студент',
            r'(\d+)\s*официант',
        ]
        
        for pattern in crowd_patterns:
            match = re.search(pattern, scene_text, re.IGNORECASE)
            if match:
                try:
                    count = int(match.group(1))
                    if 0 < count < 10000:  # Разумные пределы
                        return count
                except (ValueError, TypeError):
                    continue
        
        return 0
    
    def _validate_costumes(self, costumes: List[str], scene_text: str) -> List[str]:
        """Валидация костюмов - только конкретные описания"""
        if not costumes:
            return costumes
        
        valid_costumes = []
        costume_keywords = ['костюм', 'платье', 'одежда', 'форма', 'униформа', 'наряд']
        
        for costume in costumes:
            costume_str = str(costume).strip().lower()
            if not costume_str:
                continue
            
            # Проверяем, что это действительно описание костюма
            if any(keyword in costume_str for keyword in costume_keywords):
                valid_costumes.append(costume)
            elif len(costume_str) > 5:  # Если достаточно длинное описание
                # Проверяем, что это не просто общее слово
                if costume_str not in ['одежда', 'костюм', 'форма']:
                    valid_costumes.append(costume)
        
        return valid_costumes

