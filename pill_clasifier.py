# %%
from torchsummary import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms.v2 import Resize, RandomRotation, Compose, RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import torchvision.models as models
from matplotlib import pyplot as plt
import data_utils as du
from ultralytics import YOLO
from tqdm import tqdm

# %% [markdown]
# Загружаем изображения

# %%
dataset_path = '/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/ogyeiv2'
train_dataset = ImageFolder(dataset_path + '/train')
val_dataset = ImageFolder(dataset_path + '/test')

# %%
assert len(train_dataset.classes) == len(val_dataset.classes)
classes = train_dataset.classes
num_classes = len(classes)

for dataset in [train_dataset, val_dataset]:
    print(f"{len(dataset)} images of classes: {dataset.classes}")

# %% [markdown]
# Посмотрим примеры изображений

# %%
fig = plt.figure(figsize=(30, 10))
for index in range(1, 3):
    image, label = train_dataset[index]
    print(classes[label])
    plt.subplot(1, 10, index)
    plt.imshow(image, cmap='gray')

# %% [markdown]
# Видим, что изображения имеют высокое разрешение, порядка 8Мп.
#
# Большая часть изображения это однородный фон, который не содержит никаких полезных фич. \
# Полезных метаданных (например, centercrop, в который гарантированно поместится таблетка) по этому датасету я не нашёл, поэтому начал искать примеры статей на тему распознавания таблеток.
#
# Нашёл вот эту:
# https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.13273
#
# Там датасет проходил препроцессинг с помощью YOLO: вырезали ббокс с таблеткой и сохраняли только его. Поступим так же.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5-6 минут
mean, std = du.clean_up(
    train_dataset, device,
    "/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/ogyeiv2_clean/train")

if mean and std:
    print(f"Data set mean: {mean}, std: {std}")
else:
    # Значения статические, посчитал на своей машине.
    # Во время проверки на вашей машине эти же значения должна вернуть ф-ия
    # очистки датасета.
    mean = [0.717, 0.644, 0.605]
    std = [0.110, 0.103, 0.103]


du.clean_up(
    val_dataset, device,
    "/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/ogyeiv2_clean/test")

# Загружаем уже очищенные данные
dataset_path = '/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/ogyeiv2_clean'
train_dataset = ImageFolder(dataset_path + '/train')
val_dataset = ImageFolder(dataset_path + '/test')

# %% [markdown]
# Создадим датасет и загрузчик.

# %%
target_w = 224
target_h = 224

train_transforms = Compose([
    Resize((target_w, target_h)),
    Normalize(mean, std),
    RandomRotation([-5, 5], fill=255.),
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5)
])

val_transforms = Compose([
    Resize((target_w, target_h)),
    Normalize(mean, std),
])

# %% [markdown]
# Используем в качестве классификатора модифицированную модель EfficientNet.
#
# Заменим полносвязный слой, т. к. нам хватит 84 классов и дообучим.

# %%
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')

# Посмотрим на summary модели, чтобы найти какой слой надо заменить
summary(model, input_size=(3, target_h, target_w), device='cpu')

# %% [markdown]
# Замораживаем слои, подменяем последний слой, размораживаем его

# %%
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Linear(
    in_features=576,
    out_features=num_classes)

for param in model.classifier.parameters():
    param.requires_grad = True

summary(model, input_size=(3, target_h, target_w), device='cpu')

# %% [markdown]
# Отправляем модель на GPU для обучения

# %%
model.to(device)

# %%
# Очень маленький размер батча для GPU.
# Качество обучения сильно зависит от этого параметра.
# Полагаю, дело в том что датасет крайне мал. Всего 28 изображений на класс.
batch_size = 16

train_dataset = du.OgyeivDataset(
    train_dataset,
    train_transforms,
    device)

val_dataset = du.OgyeivDataset(
    val_dataset,
    val_transforms,
    device)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


def train_one_epoch(epoch_index):
    running_loss = 0.
    avg_loss = 0.

    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss = running_loss / (i + 1)

    return avg_loss


# %%
# Почему столько ?
# Подобрано экспериментально. При большем числе эпох ошибка уже  не уменьшается.
EPOCHS = 20
best_vloss = 1e5

for epoch in range(EPOCHS):
    print(f'Эпоха {epoch}')

    model.train(True)
    avg_loss = train_one_epoch(epoch)

    model.eval()
    running_vloss = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            running_vloss += criterion(voutputs, vlabels)

    avg_vloss = running_vloss / (i + 1)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'pill_classifier_{epoch}.pt'
        torch.save(model.state_dict(), model_path)

    print(f'Ошибка обучения: {avg_loss}, ошибка валидации: {avg_vloss}')

# %% [markdown]
# Возвращаем модель на CPU, строим отчёт

# %%
model.to('cpu')
model.eval()
labels_predicted = []
labels_true = []

with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to('cpu'), labels.to('cpu')

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        labels_predicted.extend(predicted.numpy())
        labels_true.extend(labels.numpy())

print(classification_report(labels_true, labels_predicted, target_names=classes))
