#!/usr/bin/env python3
"""
main.py

Простой локальный пайплайн для разбора сценария на сцены и извлечения метаданных.
Выход: Excel/CSV с полями для pre-production.

Требования:
  pip install transformers torch pandas openpyxl pdfplumber python-docx regex tqdm

Как использовать:
  python main.py --input path/to/script.pdf --out outputs/preproduction_table.xlsx

Параметры модели:
  --model MODEL_NAME   (по умолчанию: "gpt2" — только для теста; замените на оффлайн модель типа "mistralai/Mistral-7B-Instruct-v0.2")
  --device DEVICE      (cpu или cuda)

Если модель не найдена локально, скрипт попытается скачать через transformers (нужен интернет).

Автор: сгенерировано ChatGPT
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any

import pandas as pd

# Внешние зависимости: pdfplumber, python-docx
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docx import Document
except Exception:
    Document = None

# transformers lazy import (не падать при отсутствии)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

from tqdm import tqdm


# -----------------------------
#  I/O: чтение разных форматов
# -----------------------------

def read_file(path: str) -> str:
    """Читает .pdf, .docx, .txt (utf-8) и возвращает текст."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        if pdfplumber is None:
            raise RuntimeError("pdfplumber не установлен. Установите: pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)
    elif ext == ".docx":
        if Document is None:
            raise RuntimeError("python-docx не установлен. Установите: pip install python-docx")
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())
    else:
        # txt or fallback
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


# -----------------------------
#  Сегментация на сцены
# -----------------------------

def split_scenes(script_text: str, min_len: int = 80) -> List[str]:
    """Разбивает сценарий на сцены используя эвристику по меткам INT./EXT./СЦЕНА/ИНТ./ЭКСТ.
    Возвращает список сцен (строк).
    """
    # Нормализуем переносы
    text = script_text.replace('\r\n', '\n').replace('\r', '\n')

    # Шаблон для начала сцены — учитываем английские и русские обозначения
    pattern = r"(?=(?:^|\n)(?:INT\.|EXT\.|ИНТ\.|ЭКСТ\.|СЦЕНА\s*\d+|Scene\s*\d+))"
    chunks = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

    scenes = [c.strip() for c in chunks if c and len(c.strip()) >= min_len]
    # Если не нашлось явных меток, разбиваем по двойным переводам строки
    if len(scenes) <= 1:
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) >= min_len]
        scenes = parts

    return scenes


# -----------------------------
#  Модель: загрузка и вызов
# -----------------------------

def load_model(model_name: str = "gpt2", device: str = "cpu"):
    """Загрузить модель через transformers pipeline.
    Возвращает callable pipeline(text) -> str (response text).

    Для оффлайн-использования: укажите локальную модель/путь.
    """
    if pipeline is None:
        raise RuntimeError("transformers не установлен. Установите: pip install transformers torch")

    # Используем text-generation pipeline
    kwargs = {"model": model_name}
    try:
        if device == "cuda":
            gen = pipeline("text-generation", model=model_name, device=0)
        else:
            gen = pipeline("text-generation", model=model_name, device=-1)
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель {model_name}: {e}")

    def generate(prompt: str, max_new_tokens: int = 256) -> str:
        out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
        # out: list of dicts with 'generated_text'
        if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
            return out[0]["generated_text"]
        return str(out)

    return generate


# -----------------------------
#  Промпт и обработка сцены
# -----------------------------

PROMPT_TEMPLATE = (
    "Ты — ассистент продюсера. Извлеки из описания сцены данные в JSON с полями:\n"
    "- location (например: квартира, улица, офис)\n"
    "- time_of_day (утро/день/вечер/ночь/не указано)\n"
    "- main_characters (список имён)\n"
    "- extras (массовка, если упомянуто)\n"
    "- props (реквизит)\n"
    "- special_fx (спецэффекты, дым, взрыв и т.п.)\n"
    "Отвечай ТОЛЬКО JSON. Если не уверен — ставь пустой список или пустую строку.\n\n"
    "Сцена:\n\"{scene}\"\n\nОтвет JSON:"
)


def analyze_scene_with_model(generate_fn, scene_text: str) -> Dict[str, Any]:
    """Промпт к модели и парсинг JSON из ответа.
    В случае ошибки парсинга возвращаем 'raw_output'.
    """
    prompt = PROMPT_TEMPLATE.format(scene=scene_text)
    raw = generate_fn(prompt, max_new_tokens=512)

    # Извлекаем JSON из текста — ищем первую фигурную скобку и пытаемся загрузить
    try:
        json_start = raw.index("{")
        json_text = raw[json_start:]
        # Поправим возможные хвосты: отрежем до последней фигурной
        # Найдём соответствие скобок — простая эвристика
        stack = 0
        end_idx = None
        for i, ch in enumerate(json_text):
            if ch == "{":
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    end_idx = i + 1
                    break
        if end_idx is not None:
            json_text = json_text[:end_idx]
        parsed = json.loads(json_text)
        return parsed
    except Exception:
        # Если не получилось — возвращаем необработанный текст для диагностики
        return {"raw_output": raw}


# -----------------------------
#  Сбор в таблицу и экспорт
# -----------------------------

def build_table(scene_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, res in enumerate(scene_results, start=1):
        row = {
            "scene_number": i,
            "location": res.get("location") if isinstance(res, dict) else None,
            "time_of_day": res.get("time_of_day") if isinstance(res, dict) else None,
            "main_characters": ", ".join(res.get("main_characters", [])) if isinstance(res, dict) else None,
            "extras": ", ".join(res.get("extras", [])) if isinstance(res, dict) else None,
            "props": ", ".join(res.get("props", [])) if isinstance(res, dict) else None,
            "special_fx": ", ".join(res.get("special_fx", [])) if isinstance(res, dict) else None,
            "raw_output": res.get("raw_output") if isinstance(res, dict) else None,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# -----------------------------
#  CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Scene extraction pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to script (.pdf/.docx/.txt)")
    parser.add_argument("--out", "-o", default="preproduction_table.xlsx", help="Output XLSX or CSV")
    parser.add_argument("--model", "-m", default=os.environ.get("SCENE_MODEL", "gpt2"), help="Model name or local path for transformers")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-scenes", type=int, default=0, help="Максимум сцен для обработки (0 = все)")
    parser.add_argument("--min-scene-len", type=int, default=80, help="Мин. длина блока при сегментации")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Входной файл не найден: {args.input}")
        sys.exit(2)

    print("Чтение файла...")
    text = read_file(args.input)
    print(f"Общий объём текста: {len(text)} символов")

    print("Сегментация на сцены...")
    scenes = split_scenes(text, min_len=args.min_scene_len)
    print(f"Найдено сцен: {len(scenes)}")
    if args.max_scenes and args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    print(f"Загрузка модели: {args.model} (device={args.device})")
    try:
        generate_fn = load_model(args.model, device=args.device)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print("Выход.")
        sys.exit(3)

    results = []
    print("Анализ сцен...")
    for s in tqdm(scenes, desc="scenes"):
        parsed = analyze_scene_with_model(generate_fn, s)
        results.append(parsed)

    print("Формирование таблицы...")
    df = build_table(results)

    out = args.out
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)
    print(f"Экспорт завершён: {out}")


if __name__ == "__main__":
    main()
