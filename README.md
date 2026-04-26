# Ускорение инференса UNet модели

В данном репозитории содержатся эксперименты по ускорению инференса UNet‑подобных моделей для сегментации изображений.

## Структура репозитория

- initial_experiments.ipynb — Jupyter Notebook с первыми экспериментами (FP16 baseline, torch.compile, ONNX Runtime, квантизация, спарсификация).
- utils.py — вспомогательные функции для замера латентности, вычисления IoU и загрузки датасета.
- train_unet.py — скрипт для полноценного обучения модели на PASCAL VOC 2012.
- experiments_description.pdf — подробное описание проекта, план экспериментов и ожидаемые результаты.
- requirements.txt — список зависимостей для воспроизведения окружения.
- baseline_fp16.csv, compiled.csv, onnx_runtime.csv, ptq.csv, sparse.csv, full_comparison.csv — результаты замеров латентности (генерируются при выполнении ноутбука).
- latency_comparison.png, all_methods_comparison.png — графики сравнения.

## Установка и запуск

1. Клонируйте репозиторий:
   git clone https://github.com/RomanLomovsky/unet-acceleration.git
   cd unet-acceleration

2. Создайте виртуальное окружение и установите зависимости:
   python -m venv unet_env
   source unet_env/bin/activate        # Linux/Mac
   unet_env\Scripts\activate           # Windows
   pip install -r requirements.txt

3. Зарегистрируйте ядро Jupyter (опционально):
   python -m ipykernel install --user --name=unet_env --display-name="UNet Acceleration"

4. Запустите Jupyter Notebook:
   jupyter notebook initial_experiments.ipynb

5. Для получения осмысленного IoU обучите модель:
   python train_unet.py

## Результаты экспериментов

Примерные результаты на GPU NVIDIA T4:

Batch size | FP16 (ms) | torch.compile (ms) | ONNX Runtime (ms) | PTQ (ms) | Sparsity (ms)
1          | 7.66      | 7.43               | 7.50              | 7.20     | 7.60
2          | 7.41      | 6.94               | 7.10              | 6.80     | 7.35
4          | 9.37      | 9.24               | 8.90              | 8.50     | 9.20
8          | 17.45     | 16.39              | 16.10             | 15.30    | 17.00
16         | 31.81     | 28.58              | 28.20             | 26.50    | 30.50
32         | 61.12     | 54.00              | 53.50             | 49.00    | 58.00

Наблюдения:
- torch.compile даёт ускорение до 13% при больших batch size.
- ONNX Runtime показывает близкие к torch.compile результаты.
- Квантизация (PTQ) ускоряет до 20%, но может снизить IoU на 1–2%.
- Спарсификация без специального аппаратного ускорения почти не даёт прироста.

## Команда

Ломовский Роман
Telegram: @rustam_shaimanov