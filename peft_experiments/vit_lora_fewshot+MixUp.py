import os
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import math

TRAIN_DIR = "/home/meixuan/data/train_few_shot"
MODEL_NAME = 'vit_large_patch16_224'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
N_AUG = 20
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
LORA_R = 4
LORA_ALPHA = 32
MIXUP_ALPHA = 0.2
PREFIX = "ViT_L_LoRA_MixUp_"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=LORA_R, lora_alpha=LORA_ALPHA, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.scaling = lora_alpha / r
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def inject_lora_to_vit(model):
    for block in model.blocks:
        old_qkv = block.attn.qkv
        new_qkv = LoRALinear(old_qkv.in_features, old_qkv.out_features, bias=old_qkv.bias is not None)
        new_qkv.linear.weight.data = old_qkv.weight.data.clone()
        if old_qkv.bias is not None:
            new_qkv.linear.bias.data = old_qkv.bias.data.clone()
        block.attn.qkv = new_qkv

        old_proj = block.attn.proj
        new_proj = LoRALinear(old_proj.in_features, old_proj.out_features, bias=old_proj.bias is not None)
        new_proj.linear.weight.data = old_proj.weight.data.clone()
        if old_proj.bias is not None:
            new_proj.linear.bias.data = old_proj.bias.data.clone()
        block.attn.proj = new_proj

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name or 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def build_lora_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=len(CLASS_NAMES))
    inject_lora_to_vit(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数数量: {trainable:,}")
    return model

class MixUpDataset(Dataset):
    def __init__(self, root_dir, class_names, transform, n_aug=N_AUG, mixup_alpha=MIXUP_ALPHA):
        self.samples = []
        self.transform = transform
        self.n_aug = n_aug
        self.mixup_alpha = mixup_alpha
        for cls_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir): continue
            for fname in os.listdir(class_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), cls_idx))
        self.n_origin = len(self.samples)
        self.class_pools = {}

    def set_class_pools(self, train_origin_indices):
        self.class_pools = {i: [] for i in range(len(CLASS_NAMES))}
        for idx in train_origin_indices:
            img_path, label = self.samples[idx]
            self.class_pools[label].append(img_path)

    def __len__(self):
        return self.n_origin * (1 + self.n_aug)

    def __getitem__(self, idx):
        if idx < self.n_origin:
            img_path, label = self.samples[idx]
            return self.transform(Image.open(img_path).convert('RGB')), label
        idx2 = idx - self.n_origin
        base_idx = idx2 // self.n_aug
        img_path, label = self.samples[base_idx]
        img1 = Image.open(img_path).convert('RGB')
        other = random.choice(self.class_pools[label])
        img2 = Image.open(other).convert('RGB')
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return lam * self.transform(img1) + (1 - lam) * self.transform(img2), label

base_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def train_one_fold(train_origin_idx, train_full_idx, val_idx, dataset):
    dataset.set_class_pools(train_origin_idx)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_full_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    model = build_lora_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 3 == 0 or epoch == EPOCHS-1:
            print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            preds = model(imgs.to(DEVICE)).argmax(dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return all_preds, acc, f1

print("加载 ViT-L/16 并注入手动 LoRA...")
dataset = MixUpDataset(TRAIN_DIR, CLASS_NAMES, base_transform)
n_origin = dataset.n_origin
base_labels = [dataset[i][1] for i in range(n_origin)]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
all_val_preds, all_val_labels = [], []

for fold, (train_base, val_base) in enumerate(skf.split(np.arange(n_origin), base_labels)):
    print(f"\n===== Fold {fold+1} =====")
    train_origin = list(train_base)
    train_full = list(train_base)
    for i in train_base:
        start = n_origin + i * N_AUG
        train_full.extend(range(start, start + N_AUG))
    val_idx = list(val_base)

    preds, acc, f1 = train_one_fold(train_origin, train_full, val_idx, dataset)
    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_val_preds.extend(preds)
    all_val_labels.extend([dataset[i][1] for i in val_idx])
    print(f"  Fold {fold+1}: Accuracy={acc*100:.2f}%, Macro F1={f1:.4f}")

mean_acc = np.mean(fold_accs)
std_acc = np.std(fold_accs)
mean_f1 = np.mean(fold_f1s)
std_f1 = np.std(fold_f1s)

print(f"\n{'='*50}")
print("LoRA 微调 5 折交叉验证结果 (MixUp增强)")
print(f"Accuracy:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
print("\n详细分类报告:")
print(classification_report(all_val_labels, all_val_preds, target_names=CLASS_NAMES, digits=4))

pd.DataFrame([{
    'Method': 'ViT-L/16 + LoRA (MixUp)',
    'Mean Accuracy': f"{mean_acc*100:.2f}%",
    'Std Accuracy': f"{std_acc*100:.2f}%",
    'Mean Macro F1': f"{mean_f1:.4f}",
    'Std Macro F1': f"{std_f1:.4f}",
    'N Folds': 5
}]).to_csv(f'{PREFIX}results.csv', index=False)
print(f"\n→ {PREFIX}results.csv 已保存")
