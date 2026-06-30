"""
DINOv2 ViT-S/14 图像空间 MixUp vs 特征空间 MixUp 对比实验
增强策略：标准增强 + MixUp (α=0.4) + 高斯噪声 (0.01) + 随机擦除 (p=0.1)
分类器：RBF SVM (细粒度 C 搜索)
两种MixUp方式分别评估
严格遵守无数据泄露原则
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
)
from torchvision import transforms
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
TRAIN_DIR  = "/home/meixuan/data/train_few_shot"
CACHE_DIR  = "./dinov2_vits_cache"           # 复用已有特征缓存
OUTPUT_DIR = "./dinov2_vits_mixup_space_output"
CKPT_DIR   = "./dinov2_vits_mixup_space_ckpt"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

# ==================== 超参 ====================
INPUT_SIZE   = 224          # ViT-S 可接受 224
N_AUG        = 30
SVM_C_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
MIXUP_ALPHA  = 0.4
GAUSSIAN_STD = 0.01
ERASING_P    = 0.1
RANDOM_SEED  = 42

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")
print("DINOv2 ViT-S/14: 图像空间 MixUp vs 特征空间 MixUp 对比")

# ==================== 断点管理 ====================
def get_ckpt_path(config_name):
    return os.path.join(CKPT_DIR, f"ckpt_{config_name}.json")

def save_ckpt(config_name, data):
    with open(get_ckpt_path(config_name), 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(config_name):
    path = get_ckpt_path(config_name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

interrupted = False
def signal_handler(sig, frame):
    global interrupted
    interrupted = True
    print("\n\n⚠ 收到中断信号，正在安全退出...")
signal.signal(signal.SIGINT, signal_handler)

# ==================== 加载模型 ====================
print("加载 DINOv2 ViT-S/14...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = model.to(DEVICE).eval()

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

@torch.no_grad()
def extract_feature(img_tensor):
    out = model.forward_features(img_tensor.to(DEVICE))
    feat = out['x_norm_patchtokens'].mean(dim=1)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

# ==================== 原始特征缓存（优先复用已有）====================
def extract_original_features():
    cache_feats  = os.path.join(CACHE_DIR, "orig_feats.npy")
    cache_labels = os.path.join(CACHE_DIR, "orig_labels.npy")
    cache_paths  = os.path.join(CACHE_DIR, "orig_paths.npy")
    if os.path.exists(cache_feats) and os.path.exists(cache_labels) and os.path.exists(cache_paths):
        print("✓ 加载原图特征缓存...")
        return (np.load(cache_feats), np.load(cache_labels), np.load(cache_paths, allow_pickle=True))

    print("提取原图特征（250张）...")
    feats, labels, paths = [], [], []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        for fn in tqdm(imgs, desc=f"原图 {cls}"):
            img_path = os.path.join(cls_dir, fn)
            img = Image.open(img_path).convert('RGB')
            t = base_tf(img).unsqueeze(0)
            feat = extract_feature(t)
            feats.append(feat)
            labels.append(cls)
            paths.append(img_path)
    feats  = np.array(feats)
    labels = np.array(labels)
    paths  = np.array(paths)
    np.save(cache_feats, feats)
    np.save(cache_labels, labels)
    np.save(cache_paths, paths)
    print(f"  特征维度: {feats.shape}")
    return feats, labels, paths

# ==================== 实验一：图像空间 MixUp ====================
def run_image_space_mixup(orig_feats, orig_labels, orig_paths):
    config_name = "image_space_mixup"
    print(f"\n{'='*60}")
    print(f"实验: 图像空间 MixUp")
    print(f"{'='*60}")

    ckpt = load_ckpt(config_name)
    if ckpt and 'final_f1' in ckpt:
        print(f"  已完成，从断点恢复: Macro F1 = {ckpt['final_f1']:.4f}")
        return ckpt

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_cache = {}

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(orig_feats, orig_labels)):
        fold_seed = RANDOM_SEED + fold_id * 100
        random.seed(fold_seed)
        np.random.seed(fold_seed)

        cls_to_paths = {cls: [] for cls in CLASS_NAMES}
        for i in train_idx:
            cls_to_paths[orig_labels[i]].append(orig_paths[i])

        std_feats, mixup_feats = [], []
        std_labels, mixup_labels = [], []

        for idx in tqdm(train_idx, desc=f"折 {fold_id} 增强", leave=False):
            cls = orig_labels[idx]
            img_path = orig_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img_t = base_tf(img).unsqueeze(0)

            candidates = [p for p in cls_to_paths[cls] if p != img_path]
            if not candidates:
                candidates = cls_to_paths[cls]

            # 标准增强
            for _ in range(N_AUG):
                aug_tensor = enhanced_aug(img).unsqueeze(0)
                feat = extract_feature(aug_tensor)
                std_feats.append(feat)
                std_labels.append(cls)

            # 图像空间 MixUp
            for _ in range(N_AUG):
                other_path = random.choice(candidates)
                other_img  = Image.open(other_path).convert('RGB')
                other_t    = base_tf(other_img).unsqueeze(0)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mixed = lam * img_t + (1 - lam) * other_t
                feat = extract_feature(mixed)
                mixup_feats.append(feat)
                mixup_labels.append(cls)

        aug_feats  = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
        aug_labels = np.concatenate([np.array(std_labels), np.array(mixup_labels)])
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx':   val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")

    # C 值搜索
    cv_records = []
    best_f1 = 0.0
    best_C  = SVM_C_VALUES[0]
    for C in SVM_C_VALUES:
        fold_f1 = []
        for fold_id, cache in fold_cache.items():
            tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
            tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
            val_f = orig_feats[cache['val_idx']]
            val_l = orig_labels[cache['val_idx']]
            clf = SVC(kernel='rbf', C=C, gamma='scale', class_weight='balanced', random_state=42)
            clf.fit(tr_f, tr_l)
            pred = clf.predict(val_f)
            fold_f1.append(f1_score(val_l, pred, average='macro'))
        mf1 = np.mean(fold_f1)
        cv_records.append({'C': C, 'Macro_F1': mf1})
        if mf1 > best_f1:
            best_f1 = mf1
            best_C = C
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")

    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    return _final_evaluate_and_save(
        config_name, orig_feats, orig_labels, fold_cache,
        best_C, best_f1, cv_records, "图像空间 MixUp"
    )

# ==================== 实验二：特征空间 MixUp ====================
def run_feature_space_mixup(orig_feats, orig_labels, orig_paths):
    config_name = "feature_space_mixup"
    print(f"\n{'='*60}")
    print(f"实验: 特征空间 MixUp")
    print(f"{'='*60}")

    ckpt = load_ckpt(config_name)
    if ckpt and 'final_f1' in ckpt:
        print(f"  已完成，从断点恢复: Macro F1 = {ckpt['final_f1']:.4f}")
        return ckpt

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_cache = {}

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(orig_feats, orig_labels)):
        fold_seed = RANDOM_SEED + fold_id * 100
        random.seed(fold_seed)
        np.random.seed(fold_seed)

        cls_to_indices = {cls: [] for cls in CLASS_NAMES}
        for i in train_idx:
            cls_to_indices[orig_labels[i]].append(i)

        std_feats, mixup_feats = [], []
        std_labels, mixup_labels = [], []

        for idx in tqdm(train_idx, desc=f"折 {fold_id} 增强", leave=False):
            cls = orig_labels[idx]
            img_path = orig_paths[idx]
            img = Image.open(img_path).convert('RGB')
            feat_a = orig_feats[idx]

            candidates = [i for i in cls_to_indices[cls] if i != idx]
            if not candidates:
                candidates = cls_to_indices[cls]

            # 标准增强（在线推理）
            for _ in range(N_AUG):
                aug_tensor = enhanced_aug(img).unsqueeze(0)
                feat = extract_feature(aug_tensor)
                std_feats.append(feat)
                std_labels.append(cls)

            # 特征空间 MixUp（无需推理）
            for _ in range(N_AUG):
                other_idx = random.choice(candidates)
                feat_b = orig_feats[other_idx]
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mixed = lam * feat_a + (1 - lam) * feat_b
                norm = np.linalg.norm(mixed)
                if norm > 1e-8:
                    mixed = mixed / norm
                mixup_feats.append(mixed)
                mixup_labels.append(cls)

        aug_feats  = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
        aug_labels = np.concatenate([np.array(std_labels), np.array(mixup_labels)])
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx':   val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")

    # C 值搜索
    cv_records = []
    best_f1 = 0.0
    best_C  = SVM_C_VALUES[0]
    for C in SVM_C_VALUES:
        fold_f1 = []
        for fold_id, cache in fold_cache.items():
            tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
            tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
            val_f = orig_feats[cache['val_idx']]
            val_l = orig_labels[cache['val_idx']]
            clf = SVC(kernel='rbf', C=C, gamma='scale', class_weight='balanced', random_state=42)
            clf.fit(tr_f, tr_l)
            pred = clf.predict(val_f)
            fold_f1.append(f1_score(val_l, pred, average='macro'))
        mf1 = np.mean(fold_f1)
        cv_records.append({'C': C, 'Macro_F1': mf1})
        if mf1 > best_f1:
            best_f1 = mf1
            best_C = C
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")

    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    return _final_evaluate_and_save(
        config_name, orig_feats, orig_labels, fold_cache,
        best_C, best_f1, cv_records, "特征空间 MixUp"
    )

# ==================== 通用最终评估与保存函数 ====================
def _final_evaluate_and_save(config_name, orig_feats, orig_labels, fold_cache,
                              best_C, best_f1, cv_records, label_name):
    prefix = f"DINOv2_ViTS_{config_name}"

    # 最终评估
    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    confidence = np.empty(len(orig_labels), dtype=float)
    all_decisions = np.zeros((len(orig_labels), len(CLASS_NAMES)))

    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = SVC(kernel='rbf', C=best_C, gamma='scale', class_weight='balanced', random_state=42)
        clf.fit(tr_f, tr_l)
        val_f = orig_feats[cache['val_idx']]
        pred = clf.predict(val_f)
        dec = clf.decision_function(val_f)
        conf = np.max(dec, axis=1)
        y_pred_cv[cache['val_idx']] = pred
        confidence[cache['val_idx']] = conf
        all_decisions[cache['val_idx']] = dec

    final_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    acc = accuracy_score(orig_labels, y_pred_cv)
    print(f"  最终 CV: Acc={acc*100:.2f}%  Macro F1={final_f1:.4f}")

    # ----- 保存 C 值搜索记录 -----
    pd.DataFrame(cv_records).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_cv_f1.csv'), index=False)

    # ----- 保存混淆矩阵 -----
    cm = confusion_matrix(orig_labels, y_pred_cv, labels=CLASS_NAMES)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confusion_counts.csv'))
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confusion_normalized.csv'))

    # ----- 保存分类报告 -----
    p, r, f1_per, s = precision_recall_fscore_support(
        orig_labels, y_pred_cv, labels=CLASS_NAMES, average=None)
    rows = [{'Category': cls, 'Precision': p[i], 'Recall': r[i],
             'F1-score': f1_per[i], 'Support': s[i]} for i, cls in enumerate(CLASS_NAMES)]
    mp, mr, mf, _ = precision_recall_fscore_support(
        orig_labels, y_pred_cv, labels=CLASS_NAMES, average='macro')
    rows.append({'Category': 'Macro Avg', 'Precision': mp,
                 'Recall': mr, 'F1-score': mf, 'Support': len(orig_labels)})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_classification_report.csv'), index=False)

    # ----- 保存错误案例 -----
    errors = []
    for i in range(len(orig_labels)):
        if orig_labels[i] != y_pred_cv[i]:
            errors.append({'index': i, 'true': orig_labels[i], 'pred': y_pred_cv[i],
                           'confidence': confidence[i]})
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    pd.DataFrame(errors[:10]).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_error_cases.csv'), index=False)

    # ----- 保存置信度分布 -----
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0, float('inf')]
    labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0', '>1.0']
    conf_dist = pd.cut(confidence, bins=bins, labels=labels, right=False)
    conf_stats = conf_dist.value_counts().sort_index()
    pd.DataFrame({'Confidence Range': conf_stats.index, 'Count': conf_stats.values}).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confidence_distribution.csv'), index=False)

    # ----- 保存决策函数均值 -----
    df_dec = pd.DataFrame(all_decisions, columns=CLASS_NAMES)
    df_dec['True'] = orig_labels
    dec_mean = df_dec.groupby('True')[CLASS_NAMES].mean()
    dec_mean.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_decision_mean.csv'))

    # ----- 终端打印 -----
    print("\n归一化混淆矩阵:")
    print("           " + "  ".join(f"{c:>8}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        row_str = "  ".join(f"{cm_norm[i, j]:.4f}" for j in range(len(CLASS_NAMES)))
        print(f"{cls:>8}  {row_str}")

    print(f"\n{'Category':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
    print("-" * 55)
    for row in rows:
        print(f"{row['Category']:<12} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
              f"{row['F1-score']:>10.4f} {row['Support']:>10}")

    print(f"\n{'#':<4} {'True':<12} {'Predicted':<12} {'Confidence':>10}")
    print("-" * 42)
    for i, err in enumerate(errors[:5]):
        print(f"{i+1:<4} {err['true']:<12} {err['pred']:<12} {err['confidence']:>10.4f}")

    print(f"  图表和 CSV 已保存到 {OUTPUT_DIR}/")

    # 保存断点
    result = {
        'best_C': best_C,
        'best_cv_f1': best_f1,
        'final_f1': final_f1,
        'final_acc': acc
    }
    save_ckpt(config_name, result)
    return result

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()

# 运行图像空间 MixUp
result_image = run_image_space_mixup(orig_feats, orig_labels, orig_paths)

# 运行特征空间 MixUp
result_feature = run_feature_space_mixup(orig_feats, orig_labels, orig_paths)

# ==================== 汇总对比 ====================
print("\n" + "="*60)
print("汇总: DINOv2 ViT-S/14 图像空间 MixUp vs 特征空间 MixUp")
print("="*60)

f1_image   = result_image['final_f1']   if result_image   else float('nan')
f1_feature = result_feature['final_f1'] if result_feature else float('nan')

print(f"图像空间 MixUp: Macro F1 = {f1_image:.4f}")
print(f"特征空间 MixUp: Macro F1 = {f1_feature:.4f}")
print(f"差异: {f1_image - f1_feature:+.4f}")

# 保存汇总 CSV
summary_df = pd.DataFrame([
    {'特征提取器': 'DINOv2 ViT-S/14', 'MixUp方式': '图像空间 MixUp', 'Macro_F1': f1_image},
    {'特征提取器': 'DINOv2 ViT-S/14', 'MixUp方式': '特征空间 MixUp', 'Macro_F1': f1_feature}
])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'mixup_space_comparison_vits.csv'), index=False)
print(f"\n汇总已保存到 {OUTPUT_DIR}/mixup_space_comparison_vits.csv")
print("所有任务完成！")