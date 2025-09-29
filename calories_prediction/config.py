import os
import torch


PROJECT_ROOT = '/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_4/theme_4'
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data')


class Config:
    # для воспроизводимости
    SEED = 42

    # Девайс
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    # Гиперпараметры
    BATCH_SIZE = 16
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGR_LR = 1e-3
    EPOCHS = 30
    EPOCHS_WITHOUT_IMPROVEMENT = 5
    DROPOUT = 0.3
    HIDDEN_DIM = 256
    NUM_CLASSES = 555

    # Пути
    CSV_PATH = os.path.join(DATASET_ROOT, 'dish.csv')
    IMG_PATH = os.path.join(DATASET_ROOT, 'images')
    PTH_PATH = os.path.join(PROJECT_ROOT, 'best_model.pth')
