import os
import torch
import timm
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
MODEL_NAME = 'vit_large_patch16_224'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
NUM_CLASSES = len(CLASS_NAMES)
N_AUG = 20
INPUT_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
ADAPTER_DIM = 64
MIXUP_ALPHA = 0.2
PREFIX = "ViT_L_Adapter_MixUp_"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ==================== 数据集类（修复：class_pools在每折中动态构建，仅用训练集原图） ====================
class MixUpDataset(Dataset):
    def __init__(self, root_dir, class_names, transform, n_aug=N_AUG, mixup_alpha=MIXUP_ALPHA):
        self.samples = []           # 仅保存原图 (path, label)
        self.transform = transform
        self.n_aug = n_aug
        self.mixup_alpha = mixup_alpha
        for cls_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), cls_idx))
        self.n_origin = len(self.samples)
        self.class_pools = {}   # 将在 set_class_pools 中填充

    def set_class_pools(self, train_origin_indices):
        """根据训练集的原图索引动态构建 class_pools，确保不包含验证集图片"""
        self.class_pools = {cls_idx: [] for cls_idx in range(NUM_CLASSES)}
        for idx in train_origin_indices:
            img_path, label = self.samples[idx]
            self.class_pools[label].append(img_path)

    def __len__(self):
        return self.n_origin * (1 + self.n_aug)

    def __getitem__(self, idx):
        if idx < self.n_origin:
            # 原图
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            return tensor, label
        else:
            # MixUp 增强样本
            idx2 = idx - self.n_origin
            base_idx = idx2 // self.n_aug          # 属于哪张原图
            img_path, label = self.samples[base_idx]
            img1 = Image.open(img_path).convert('RGB')
            # 仅从训练集原图中随机选取配对图片
            other_path = random.choice(self.class_pools[label])
            img2 = Image.open(other_path).convert('RGB')
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            img1_tensor = self.transform(img1)
            img2_tensor = self.transform(img2)
            mixed_tensor = lam * img1_tensor + (1 - lam) * img2_tensor
            return mixed_tensor, label

base_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ==================== Adapter 模块 ====================
class Adapter(nn.Module):
    def __init__(self, dim, reduction_dim=ADAPTER_DIM):
        super().__init__()
        self.down = nn.Linear(dim, reduction_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(reduction_dim, dim)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

def inject_adapters(model):
    for block in model.blocks:
        out_dim = block.mlp.fc2.out_features
        block.mlp = nn.Sequential(block.mlp, Adapter(out_dim))
    return model

def build_adapter_model():
    base_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
    base_model = inject_adapters(base_model)
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.head.parameters():
        param.requires_grad = True
    for name, param in base_model.named_parameters():
        if 'mlp.1' in name:
            param.requires_grad = True
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    print(f"  可训练参数数量: {trainable_params:,}")
    return base_model

# ==================== 单折训练与评估 ====================
def train_one_fold(train_origin_idx, train_full_idx, val_indices, dataset):
    # 用训练集原图索引构建 class_pools
    dataset.set_class_pools(train_origin_idx)

    train_subset = torch.utils.data.Subset(dataset, train_full_idx)
    val_subset   = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_adapter_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 3 == 0 or epoch == EPOCHS - 1:
            print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return all_preds, acc, macro_f1

# ==================== 5 折交叉验证 ====================
print(f"加载 ViT-L/16 预训练模型并注入 Adapter...")
full_dataset = MixUpDataset(TRAIN_DIR, CLASS_NAMES, base_transform, n_aug=N_AUG)
n_origin = full_dataset.n_origin

base_labels = [full_dataset[i][1] for i in range(n_origin)]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs, fold_f1s = [], []
all_val_preds, all_val_labels = [], []

print(f"\n开始 5 折 Adapter 微调 (MixUp增强, bottleneck_dim={ADAPTER_DIM}, epochs={EPOCHS})...")
for fold, (train_base_idx, val_base_idx) in enumerate(skf.split(np.arange(n_origin), base_labels)):
    print(f"\n===== Fold {fold+1} =====")
    # 训练集原图索引（用于构建 class_pools）
    train_origin_idx = list(train_base_idx)
    # 训练集完整索引（原图 + 增强样本）
    train_full_idx = list(train_base_idx)
    for i in train_base_idx:
        start = n_origin + i * N_AUG
        train_full_idx.extend(range(start, start + N_AUG))
    val_idx = list(val_base_idx)  # 验证集仅用原图

    preds, acc, f1 = train_one_fold(train_origin_idx, train_full_idx, val_idx, full_dataset)
    fold_accs.append(acc)
    fold_f1s.append(f1)
    all_val_preds.extend(preds)
    all_val_labels.extend([full_dataset[i][1] for i in val_idx])
    print(f"  Fold {fold+1}: Accuracy={acc*100:.2f}%, Macro F1={f1:.4f}")

# ==================== 汇总结果 ====================
mean_acc = np.mean(fold_accs)
std_acc  = np.std(fold_accs)
mean_f1  = np.mean(fold_f1s)
std_f1   = np.std(fold_f1s)

print(f"\n{'='*50}")
print(f"Adapter 微调 5 折交叉验证结果 (MixUp增强)")
print(f"{'='*50}")
print(f"Accuracy:  {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
print(f"\n详细分类报告 (全部验证集预测汇总):")
print(classification_report(all_val_labels, all_val_preds, target_names=CLASS_NAMES, digits=4))

results = [{
    'Method': f'ViT-L/16 + Adapter (MixUp, dim={ADAPTER_DIM})',
    'Mean Accuracy': f"{mean_acc*100:.2f}%",
    'Std Accuracy': f"{std_acc*100:.2f}%",
    'Mean Macro F1': f"{mean_f1:.4f}",
    'Std Macro F1': f"{std_f1:.4f}",
    'N Folds': 5,
    'Epochs': EPOCHS,
    'Backbone': MODEL_NAME,
    'Augmentation': 'MixUp (α=0.2, 20x)'
}]
pd.DataFrame(results).to_csv(f'{PREFIX}results.csv', index=False)
print(f"\n→ {PREFIX}results.csv 已保存")
print("完成！")