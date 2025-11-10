# ScenePrepAI — сервис парсинга сценариев

Небольшой локальный сервис для разбора сценариев (.docx / .pdf) и генерации таблицы для препродакшна (КПП).
Приложение умеет:
- Разбивать сценарий на сцены
- Извлекать локации, персонажей, реквизит, массовку и другие производственные элементы
- Опционально улучшать метаданные с помощью локальной LLM (через `llama-cpp-python` и GGUF-модель)
- Экспортировать результаты в JSON / CSV / XLSX

В этом репозитории вы найдёте:
- `screenplay_parser.py` — основной парсер и CLI (корневой файл)
- `app/server.py` — лёгкий Flask-сервер, который питает веб-интерфейс
- `preprod-enterprise.html` — фронтенд UI (в корне репо)
- `requirements.txt` — зависимости
- `setup_model.sh` — (опционально) скрипт загрузки модели

## Быстрый старт (Windows, PowerShell)

1) Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Подготовьте папки (если ещё нет):

```powershell
mkdir models, output, logs, input -Force
```

3) (Опционально) Загрузите GGUF-модель для LLM-подсказок.
	- На Windows (PowerShell):

```powershell
$u = 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
Invoke-WebRequest -Uri $u -OutFile models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

	- На macOS / Linux:

```bash
./setup_model.sh
```

4) Запустите локальный сервер (фронтенд будет доступен по адресу http://127.0.0.1:5000/ ):

```powershell
python app\server.py
```

5) Откройте браузер и перейдите на http://127.0.0.1:5000/ — загрузите файл сценария (.docx или .pdf), выберите шаблон и нажмите "Начать обработку".

6) Экспортируйте результаты кнопками CSV / XLSX / JSON.

## CLI (альтернативный запуск)

Вы можете запускать парсер напрямую как CLI (правила + опционально LLM):

```powershell
# Без LLM (быстро)
python screenplay_parser.py -i input\scenario.docx -o output\production_table.xlsx --no-llm

# С LLM (если установлена llama-cpp-python и скачана модель)
python screenplay_parser.py -i input\scenario.docx -o output\production_table.xlsx --model models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## PDF поддержка
Сервер поддерживает загрузку PDF и извлечение текста через `pdfplumber`. Для сканированных PDF (изображения) потребуется OCR (pytesseract + tesseract). Если загрузка PDF возвращает сообщение об отсутствии `pdfplumber`, установите зависимости:

```powershell
pip install pdfplumber
```

## Примечания и рекомендации
- По умолчанию LLM не обязателен — если `llama-cpp-python` или модель отсутствуют, парсер работает по эвристикам.
- Для больших моделей и продакшн-использования рекомендуем запускать парсинг в фоновом воркере (RQ/Celery) и хранить результаты в базе.
- Обратите внимание на размер GGUF-моделей — они занимают несколько гигабайт.

Если хотите, могу:
- Добавить OCR fallback для сканированных PDF
- Сделать хранение результатов по job-id (чтобы несколько пользователей могли скачивать свои файлы)
- Добавить unit-тесты и CI

---

Автор: команда ScenePrepAI

