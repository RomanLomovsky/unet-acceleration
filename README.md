# Ускорение инференса UNet модели

В данном репозитории содержатся эксперименты по ускорению инференса UNet‑подобных моделей для сегментации изображений.  

## Структура репозитория

- `initial_experiments.ipynb` — Jupyter Notebook с первыми экспериментами (FP16 baseline, torch.compile, профилирование, оценка качества).
- `utils.py` — вспомогательные функции для замера латентности, вычисления IoU и загрузки датасета.
- `experiments_description.pdf` — подробное описание проекта, план экспериментов и ожидаемые результаты.
- `requirements.txt` — список зависимостей для воспроизведения окружения.
- `baseline_fp16.csv`, `compiled.csv` — сохранённые результаты замеров латентности (генерируются при выполнении ноутбука).
- `latency_comparison.png` — график сравнения латентности (генерируется автоматически).

## Установка и запуск

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/RomanLomovsky/unet-acceleration.git
   cd unet-acceleration

2. Установить зависимости можно внутри скрипта, можно поднв виртуальное окружение с помощью requirements.txt

1. Запустите Jupyter Notebook с экспериментами:
   ```bash
   jupyter notebook initial_experiments.ipynb


## Команда

Ломовский Роман
Tg: @rustam\_shaimanov