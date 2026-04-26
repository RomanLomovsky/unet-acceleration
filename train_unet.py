"""
Скрипт для обучения UNet на PASCAL VOC 2012.
Сохраняет лучшую модель для последующего ускорения.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import os

# params
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data/VOC2012"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# transformations
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_mask = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])

train_set = VOCSegmentation(DATA_ROOT, year="2012", image_set="train", download=True,
                            transform=transform_img, target_transform=transform_mask)
val_set = VOCSegmentation(DATA_ROOT, year="2012", image_set="val", download=True,
                          transform=transform_img, target_transform=transform_mask)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", 
                 in_channels=3, classes=21).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def compute_iou(preds, targets, num_classes=21):
    iou_list = []
    for c in range(num_classes):
        intersection = ((preds == c) & (targets == c)).sum().float()
        union = ((preds == c) | (targets == c)).sum().float()
        if union > 0:
            iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0.0

best_iou = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.squeeze(1).to(DEVICE)  # (B,1,H,W) -> (B,H,W)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    iou_vals = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.squeeze(1).to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks)
            iou_vals.append(iou)
    mean_iou = np.mean(iou_vals)
    print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Val IoU = {mean_iou:.4f}")
    
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_unet_voc.pth"))
        print(f"  Saved best model with IoU {best_iou:.4f}")
    
    scheduler.step()

print("Training completed.")