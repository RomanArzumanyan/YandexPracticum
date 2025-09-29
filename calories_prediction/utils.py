import os
import random
from functools import partial

import numpy as np

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import torchmetrics

from transformers import AutoModel, AutoTokenizer

from data_utils import MultimodalDataset, collate_fn, get_transforms


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def set_requires_grad(module: nn.Module, unfreeze_pattern="", verbose=False):
    if len(unfreeze_pattern) == 0:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    pattern = unfreeze_pattern.split("|")

    for name, param in module.named_parameters():
        if any([name.startswith(p) for p in pattern]):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(
            config.TEXT_MODEL_NAME)

        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size, config.HIDDEN_DIM)

        self.image_proj = nn.Linear(
            self.image_model.num_features, config.HIDDEN_DIM)

        self.regression = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_features = self.text_model(
            input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        fused_emb = text_emb * image_emb

        logits = self.regression(fused_emb)
        return logits


def train(config):
    device = config.DEVICE
    seed_everything(config.SEED)

    # Инициализация модели
    model = MultimodalModel(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(model.text_model,
                      unfreeze_pattern=config.TEXT_MODEL_UNFREEZE, verbose=True)
    set_requires_grad(model.image_model,
                      unfreeze_pattern=config.IMAGE_MODEL_UNFREEZE, verbose=True)

    # Оптимизатор с разными LR
    optimizer = AdamW([{
        'params': model.text_model.parameters(),
        'lr': config.TEXT_LR
    }, {
        'params': model.image_model.parameters(),
        'lr': config.IMAGE_LR
    }, {
        'params': model.regression.parameters(),
        'lr': config.REGR_LR
    }])

    # TODO: Можно ли использовать несколько разных лоссов ?
    # Для МАЕ, насколько я понимаю, подходит L1.
    criterion = nn.L1Loss()

    # Загрузка данных
    transforms = get_transforms(config)
    val_transforms = get_transforms(config, ds_type='test')
    train_dataset = MultimodalDataset(config, transforms)
    val_dataset = MultimodalDataset(config, val_transforms, ds_type='test')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    # инициализируем метрику
    mae_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_val = torchmetrics.MeanAbsoluteError().to(device)
    best_mae_val = 1e5
    epochs_without_improvement = 0

    print("training started")
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Подготовка данных
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            labels = torch.unsqueeze(batch['label'].to(device), 1)

            # Forward
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Умножаем на массу чтобы получить МАЕ в абсолютных величинах
            predicted = logits
            _ = mae_train(
                preds=predicted * inputs['mass'],
                target=labels * inputs['mass'])

        # Валидация
        train_mae = mae_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, mae_val)
        mae_val.reset()
        mae_train.reset()

        print(
            f"Epoch {epoch}/{config.EPOCHS-1} | avg_Loss: {total_loss/len(train_loader):.4f} | Train MAE: {train_mae:.4f}| Val MAE: {val_mae:.4f}"
        )

        if val_mae < best_mae_val:
            print(f"New best model, epoch: {epoch}")
            best_mae_val = val_mae
            torch.save(model.state_dict(), config.PTH_PATH)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement == config.EPOCHS_WITHOUT_IMPROVEMENT:
            print(
                f"Learning has reached plateau of {epochs_without_improvement} epochs without improvement.")
            break


def validate(model, val_loader, device, metric):
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            labels = torch.unsqueeze(batch['label'].to(device), 1)

            logits = model(**inputs)
            predicted = logits
            _ = metric(preds=predicted * inputs['mass'],
                       target=labels * inputs['mass'])

    return metric.compute().cpu().numpy()


def inference(config):
    device = config.DEVICE
    model = MultimodalModel(config).to(device)
    state_dict = torch.load(config.PTH_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    val_transforms = get_transforms(config, ds_type='test')
    val_dataset = MultimodalDataset(config, val_transforms, ds_type='test')
    val_loader = DataLoader(val_dataset,
                            batch_size=config.BATCH_SIZE,
                            shuffle=False,
                            collate_fn=partial(collate_fn,
                                               tokenizer=tokenizer))

    result = {}
    with torch.no_grad():
        for batch in tqdm(val_loader):

            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'image': batch['image'].to(device),
                'mass': batch['mass'].to(device),
            }
            ids = batch['id']
            val_true = (batch['label'].cpu() * inputs['mass'].cpu()).numpy()
            val_pred = (torch.squeeze(model(**inputs), dim=1).cpu() * inputs['mass'].cpu()).numpy()

            for i in range(0, len(ids)):
                result[ids[i]] = val_true[i], val_pred[i], abs(
                    val_true[i] - val_pred[i])

    return result
