# Миграция на Qwen2.5-14B для 24GB VRAM

## Что изменилось

Проект был оптимизирован для работы на GPU с 24GB VRAM (вместо изначальных 80GB).

### Основные изменения:

1. **Модель**: Mixtral-8x7B (~47GB) → Qwen2.5-14B (~8-9GB)
2. **Batch size**: 1 → 4 (можно увеличить до 6)
3. **Max tokens**: 1024 → 512
4. **Длина текста сцены**: 2000 → 1000 символов
5. **Добавлена очистка памяти** после каждого батча
6. **Добавлен мониторинг памяти** GPU

---

## Установка

### 1. Установите зависимости

```bash
# Активируйте виртуальное окружение
source .venv/bin/activate  # Linux/Mac
# или
.\.venv\Scripts\Activate.ps1  # Windows

# Установите vLLM (ОБЯЗАТЕЛЬНО для оптимальной работы)
pip install vllm

# Убедитесь что все зависимости установлены
pip install -r requirements.txt
```

### 2. Проверьте конфигурацию

Файл [`config.yaml`](config.yaml:156) уже обновлен:

```yaml
llm:
  model_name: "Qwen/Qwen2.5-14B-Instruct"
  use_vllm: true
  device: "cuda"
  generation:
    max_new_tokens: 512
    temperature: 0.05
  batch:
    entity_extraction_batch_size: 4
```

---

## Использование

### CLI

```bash
# Базовый запуск
python app/screenplay_parser.py -i Examples/scenario.docx -o output/result.xlsx

# С выбором пресета
python app/screenplay_parser.py -i Examples/scenario.docx -o output/result.xlsx --preset basic
python app/screenplay_parser.py -i Examples/scenario.docx -o output/result.xlsx --preset extended
python app/screenplay_parser.py -i Examples/scenario.docx -o output/result.xlsx --preset full
```

### Веб-интерфейс

```bash
# Запустите сервер
python app/server.py

# Откройте браузер
# http://127.0.0.1:5000/
```

---

## Ожидаемое использование памяти

### С Qwen2.5-14B-Instruct + vLLM:

- **Модель**: ~8-9GB
- **Промпты + генерация (batch=4)**: ~6-8GB
- **Пиковое использование**: ~14-17GB
- **Запас**: ~7-10GB

### Без vLLM (transformers):

- **Модель**: ~8-9GB
- **Промпты + генерация (batch=4)**: ~8-10GB
- **Пиковое использование**: ~16-19GB
- **Запас**: ~5-8GB

---

## Производительность

### Ожидаемая скорость:

- **С vLLM**: ~3-4 сцены/минуту
- **Без vLLM**: ~2-3 сцены/минуту

### Для сценария на 50 сцен:

- **С vLLM**: ~12-17 минут
- **Без vLLM**: ~17-25 минут

---

## Решение проблем

### Проблема: "CUDA out of memory"

**Решение 1**: Уменьшите batch size в [`config.yaml`](config.yaml:175):

```yaml
batch:
  entity_extraction_batch_size: 2  # Было 4
```

**Решение 2**: Используйте меньшую модель:

```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"  # ~4-5GB вместо 8-9GB
batch:
  entity_extraction_batch_size: 8  # Можно увеличить
```

**Решение 3**: Уменьшите длину текста сцены в [`llm_engine.py`](app/llm_engine.py:694):

```python
scene_text_limited = scene.scene_text[:500]  # Было 1000
```

### Проблема: vLLM не устанавливается

**Решение**: Используйте transformers (медленнее, но работает):

```yaml
use_vllm: false
```

### Проблема: Модель загружается слишком долго

**Решение**: Это нормально при первом запуске. Модель скачивается из HuggingFace (~8GB). Последующие запуски будут быстрыми.

### Проблема: Низкое качество извлечения

**Решение**: Увеличьте температуру в [`config.yaml`](config.yaml:167):

```yaml
generation:
  temperature: 0.1  # Было 0.05
```

---

## Мониторинг памяти

Логи теперь показывают использование GPU памяти:

```
INFO - GPU память до генерации: 8.45GB выделено, 9.12GB зарезервировано
INFO - Модель обработала батч за 12.3 секунд
INFO - GPU память после генерации: 8.47GB выделено, 9.12GB зарезервировано
```

Следите за этими логами. Если "выделено" приближается к 24GB - уменьшите batch size.

---

## Сравнение моделей

| Модель | VRAM | Качество | Скорость | Batch size |
|--------|------|----------|----------|------------|
| Qwen2.5-14B | ~8-9GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 4-6 |
| Qwen2.5-7B | ~4-5GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8-16 |
| Mistral-7B-v0.3 | ~4-5GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 8-16 |

**Рекомендация**: Используйте Qwen2.5-14B для лучшего качества.

---

## Альтернативные конфигурации

### Максимальное качество (медленно):

```yaml
model_name: "Qwen/Qwen2.5-14B-Instruct"
use_vllm: true
generation:
  max_new_tokens: 768
  temperature: 0.01
batch:
  entity_extraction_batch_size: 2
```

### Максимальная скорость (хуже качество):

```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
use_vllm: true
generation:
  max_new_tokens: 384
  temperature: 0.1
batch:
  entity_extraction_batch_size: 12
```

### Баланс (рекомендуется):

```yaml
model_name: "Qwen/Qwen2.5-14B-Instruct"
use_vllm: true
generation:
  max_new_tokens: 512
  temperature: 0.05
batch:
  entity_extraction_batch_size: 4
```

---

## Что исправлено

### 1. Утечка памяти в [`llm_engine.py:361-430`](app/llm_engine.py:361-430)

**Было**:
```python
outputs = self.model.generate(...)
return responses
```

**Стало**:
```python
outputs = self.model.generate(...)
# Декодирование...
del inputs
del outputs
torch.cuda.empty_cache()
gc.collect()
return responses
```

### 2. Формат промпта для Qwen

**Добавлено** в [`llm_engine.py:216-229`](app/llm_engine.py:216-229):
```python
if 'qwen' in model_name:
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
```

### 3. Мониторинг памяти

**Добавлено** в [`llm_engine.py:367-370`](app/llm_engine.py:367-370):
```python
if self.device == 'cuda':
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"GPU память: {mem_allocated:.2f}GB выделено")
```

---

## Тестирование

### Быстрый тест:

```bash
# Тест на маленьком файле
python app/screenplay_parser.py -i Examples/scenario.docx -o output/test.xlsx --preset basic
```

### Полный тест:

```bash
# Тест на полном файле
python app/screenplay_parser.py -i Examples/scenario.docx -o output/test_full.xlsx --preset full
```

### Проверка памяти:

```bash
# Во время работы в другом терминале
watch -n 1 nvidia-smi
```

---

## Поддержка

Если возникли проблемы:

1. Проверьте логи в `screenplay_parser.log`
2. Убедитесь что CUDA доступна: `python -c "import torch; print(torch.cuda.is_available())"`
3. Проверьте версию CUDA: `nvidia-smi`
4. Убедитесь что vLLM установлен: `pip show vllm`

---

## Changelog

### v2.1.0 (2025-11-16)

- ✅ Миграция на Qwen2.5-14B-Instruct
- ✅ Оптимизация для 24GB VRAM
- ✅ Исправлена утечка памяти
- ✅ Добавлен мониторинг GPU памяти
- ✅ Увеличен batch size до 4
- ✅ Добавлена поддержка vLLM
- ✅ Уменьшена длина промптов

### v2.0.0 (предыдущая версия)

- Использовала Mixtral-8x7B
- Требовала 80GB VRAM
- Batch size = 1