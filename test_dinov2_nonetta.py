"""
DINOv2 ViT-S/14 + RBF SVM 最终测试脚本（无TTA）
配置：
  模型：DINOv2 ViT-S/14
  增强：MixUp(α=0.4) + 高斯噪声(0.01) + RandomErasing(0.1)，N_AUG=30
  分类器：RBF SVM，C=0.1，gamma='scale'
  推理：单次前向，无TTA
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from torchvision import transforms
import pandas as pd
import random

# ==================== 配置 ====================
TRAIN_DIR    = "/home/meixuan/data/train_few_shot"
TEST_DIR     = "/home/meixuan/data/test/test_shuffled"
OUTPUT_CSV   = "24124053.csv"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES  = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

INPUT_SIZE   = 224
N_AUG        = 30
MIXUP_ALPHA  = 0.4
GAUSSIAN_STD = 0.01
ERASING_P    = 0.1
BEST_C       = 0.1
RANDOM_SEED  = 42

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")

# ==================== 加载 DINOv2 ViT-S/14（离线优先）====================
print("加载 DINOv2 ViT-S/14...")
hub_dir = '/home/meixuan/.cache/torch/hub/facebookresearch_dinov2_main'
weights_path = '/home/meixuan/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
if os.path.exists(hub_dir) and os.path.exists(weights_path):
    model = torch.hub.load(hub_dir, 'dinov2_vits14', pretrained=False, source='local')
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict)
else:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(DEVICE).eval()

# ==================== 数据预处理 ====================
base_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

enhanced_aug = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.Lambda(lambda x: x + GAUSSIAN_STD * torch.randn_like(x)),
    transforms.RandomErasing(p=ERASING_P, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])

# ==================== 特征提取函数 ====================
@torch.no_grad()
def extract_feature(img_tensor):
    out = model.forward_features(img_tensor.to(DEVICE))
    feat = out['x_norm_patchtokens'].mean(dim=1)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

# ==================== 全量增强训练集特征提取 ====================
def extract_all_train_features():
    all_feats, all_labels = [], []
    all_images = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        all_images.extend([os.path.join(cls_dir, f) for f in imgs])

    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        same_cls_paths = [p for p in all_images if os.path.basename(os.path.dirname(p)) == cls]

        for fn in tqdm(imgs, desc=f"训练集 {cls}"):
            img_path = os.path.join(cls_dir, fn)
            img = Image.open(img_path).convert('RGB')

            # 1) 原图
            t = base_tf(img).unsqueeze(0)
            all_feats.append(extract_feature(t))
            all_labels.append(cls)

            # 2) 标准增强 x N_AUG
            for _ in range(N_AUG):
                aug_t = enhanced_aug(img).unsqueeze(0)
                all_feats.append(extract_feature(aug_t))
                all_labels.append(cls)

            # 3) MixUp (同类内，排除自身)
            candidates = [p for p in same_cls_paths if p != img_path]
            if not candidates:
                candidates = same_cls_paths
            img_t = base_tf(img).unsqueeze(0)
            for _ in range(N_AUG):
                other_path = random.choice(candidates)
                other_img = Image.open(other_path).convert('RGB')
                other_t = base_tf(other_img).unsqueeze(0)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mixed = lam * img_t + (1 - lam) * other_t
                all_feats.append(extract_feature(mixed))
                all_labels.append(cls)

    return np.array(all_feats), np.array(all_labels)

print("提取训练集全量增强特征...")
X_train, y_train = extract_all_train_features()
print(f"训练特征矩阵形状: {X_train.shape}")

# ==================== 训练 RBF SVM ====================
print(f"训练 RBF SVM (C={BEST_C}, gamma='scale')...")
clf = SVC(kernel='rbf', C=BEST_C, gamma='scale',
          class_weight='balanced', probability=False,
          random_state=RANDOM_SEED)
clf.fit(X_train, y_train)
print("训练完成。")

# ==================== 单次推理预测（无TTA）====================
def predict_single(img_path):
    """对单张测试图片进行推理，只做基础预处理，无任何增强或TTA"""
    img = Image.open(img_path).convert('RGB')
    t = base_tf(img).unsqueeze(0)
    with torch.no_grad():
        feat = extract_feature(t)
    pred = clf.predict(feat.reshape(1, -1))[0]
    return pred

# ==================== 测试集推理 ====================
if os.path.exists(TEST_DIR):
    test_files = sorted([f for f in os.listdir(TEST_DIR)
                         if f.lower().endswith(('.png','.jpg','.jpeg'))])
    print(f"发现 {len(test_files)} 张测试图片，开始推理...")

    predictions = []
    for fname in tqdm(test_files, desc="推理中"):
        pred = predict_single(os.path.join(TEST_DIR, fname))
        predictions.append(pred)

    df = pd.DataFrame({'filename': test_files, 'label': predictions})
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ 提交文件已生成: {OUTPUT_CSV}")
    print(f"  总样本数: {len(df)}")
    print("  预测类别分布:")
    for cls in CLASS_NAMES:
        count = (df['label'] == cls).sum()
        print(f"    {cls}: {count:6d} ({100 * count / len(df):.1f}%)")
else:
    print(f"✗ 测试集目录不存在: {TEST_DIR}")