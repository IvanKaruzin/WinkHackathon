#!/usr/bin/env bash
# setup_model.sh - Скрипт для загрузки и установки модели

echo "🚀 Настройка окружения для парсера сценариев"
echo "============================================"

echo "📁 Создание структуры папок..."
mkdir -p models
mkdir -p output
mkdir -p logs
mkdir -p input

MODEL_URL="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Загрузка модели Mistral 7B (это займет время, размер несколько GB)..."
    curl -L -o "$MODEL_PATH" "$MODEL_URL"
    echo "✅ Модель загружена!"
else
    echo "✅ Модель уже существует"
fi

echo ""
echo "✨ Установка завершена!"
echo "Теперь вы можете использовать парсер:" 
echo "python screenplay_parser.py -i input/scenario.docx -o output/production.xlsx"
