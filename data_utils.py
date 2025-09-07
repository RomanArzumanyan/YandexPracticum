from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
from ultralytics import YOLO
import torch
from tqdm import tqdm
import numpy as np

model_yolo = YOLO("yolo12n.pt")


def clean_up(dataset, device, path) -> tuple[list, list]:
    if os.path.isdir(path):
        print(f"Skip cleanup because {path} already exists")
        return None, None

    model_yolo.to(device)

    mean, std_dev = [0., 0., 0.], [0., 0., 0.]
    img_idx, num_misses = 0, 0
    has_detection = False
    for x, y in tqdm(dataset):
        class_name = dataset.classes[y]
        # Ничего кроме таблеток на изображениях нет.
        # Ставим абсурдно низкий confidence, т. к. всё найденное - таблетка.
        results = model_yolo(x, conf=1e-5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                bbox = box.xyxy.cpu().numpy().reshape((4)).astype(int)
                x = x.crop(bbox)

                image = np.array(x)
                for c in range(0, 3):
                    channel = image[:, :, c]
                    mean[c] += np.mean(channel)
                    std_dev[c] += np.std(channel)

                has_detection = True
                break
            break

        if not has_detection:
            num_misses += 1

        fpath = os.path.join(path, class_name)
        os.makedirs(fpath, exist_ok=True)
        x.save(fpath + f"/image_{img_idx:06d}.png")
        img_idx += 1

    mean = [float(m / (img_idx * 255)) for m in mean]
    std_dev = [float(s / (img_idx * 255)) for s in std_dev]

    print(f"{num_misses} / {len(dataset)} detections are missing")
    return mean, std_dev


class OgyeivDataset(Dataset):
    def __init__(self, dataset, transforms, device):
        super(OgyeivDataset, self).__init__()
        self.dataset = dataset
        self.to_tensor = ToTensor()
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # x копируем в vRAM, а y -нет. Почему так?
        # GPU быстрее выполнит преобразования над изображениями, поэтому их выложим в vRAM сразу.
        # Однако, y - это просто переменная типа int, лучше её скопировать в vRAM батчем.
        x, y = self.dataset[idx]
        x = self.to_tensor(x).to(self.device)
        return self.transforms(x), y


dataset_path = '/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/ogyeiv2'
train_dataset = ImageFolder(dataset_path + '/train')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean, std = clean_up(
    train_dataset,
    device,
    "/home/vlabs/Documents/Yandex_Practicum_ML_CV/sprint_3/theme_4/lesson_1/deleteme")
print(f"mean: {mean}, std: {std}")
