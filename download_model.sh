#!/bin/bash
# Скрипт для загрузки модели Mistral 7B

MODEL_DIR="models"
MODEL_FILE="mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

echo "Загрузка модели Mistral 7B для LLM..."
echo "Размер: ~4.1 GB"
echo ""

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Модель уже существует: $MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "Начинаю загрузку..."
curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL" --progress-bar

if [ $? -eq 0 ]; then
    echo ""
    echo "Модель успешно загружена!"
    echo "Путь: $MODEL_DIR/$MODEL_FILE"
    echo ""
    echo "Теперь можно использовать LLM:"
    echo "python app/screenplay_parser.py -i input/test_scenario.docx -o output/result.xlsx"
else
    echo ""
    echo "Ошибка при загрузке модели"
    exit 1
fi