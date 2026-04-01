import torch
import torchvision
import time
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm


def measure_latency(model, input_tensor, num_runs=100, warmup=10):
    """
    Измеряет среднее время инференса (мс) для модели и входного тензора.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Замеры
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return np.mean(times) * 1000


def compute_iou(preds, targets, num_classes, ignore_index=255):
    """
    Вычисляет средний IoU для батча.
    preds, targets: тензоры (batch, H, W) с индексами классов.
    """
    iou_per_class = []
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        # исклдчаем ignore_index
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou.item())
    return np.mean(iou_per_class) if iou_per_class else 0.0


def load_voc_datasets(root="./data/VOC2012", batch_size=1, num_workers=2, download=True):
    """
    Загружает PASCAL VOC 201
    На выходе train_loader, val_loader, test_loader.
    """
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    train_set = VOCSegmentation(
        root=root, year="2012", image_set="train", download=download,
        transform=img_transform, target_transform=mask_transform
    )
    val_set = VOCSegmentation(
        root=root, year="2012", image_set="val", download=download,
        transform=img_transform, target_transform=mask_transform
    )
    # тстового набора нет, используем val как тест
    test_set = val_set

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader