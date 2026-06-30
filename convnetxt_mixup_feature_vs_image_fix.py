"""
ConvNeXt-L 特征空间 MixUp vs 图像空间 MixUp 对比实验
（同时包含纯原图多分类器对比，用于补充报告 4.2 和 4.3 节）

实验一：纯原图多分类器对比（无增强）
实验二：特征空间 MixUp（在特征向量上做线性插值）
实验三：图像空间 MixUp（在图像上做线性插值）


"""

import os
import json
import numpy as np
import torch
import timm
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from torchvision import transforms
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
TRAIN_DIR  = "/home/meixuan/data/train_few_shot"
CACHE_DIR  = "./convnext_classifier_compare_cache"
OUTPUT_DIR = "./convnext_classifier_compare_output"
CKPT_DIR   = "./convnext_classifier_compare_ckpt"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

INPUT_SIZE   = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MIXUP_ALPHA  = 0.4
N_AUG        = 30
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")

# ==================== 断点管理 ====================
def get_ckpt_path(name):
    return os.path.join(CKPT_DIR, f"ckpt_{name}.json")

def save_ckpt(name, data):
    with open(get_ckpt_path(name), 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(name):
    path = get_ckpt_path(name)
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
print("加载 ConvNeXt-L...")
model = timm.create_model('convnext_large', pretrained=True, num_classes=0)
model = model.to(DEVICE).eval()

base_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ==================== 原图特征提取 ====================
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
            t = base_tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = model(t)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().squeeze().numpy())
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

# ==================== 实验一：纯原图多分类器对比 ====================
class PrototypeClassifier:
    def __init__(self):
        self.prototypes = None
        self.classes    = None

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self.prototypes = np.stack([X[y == cls].mean(axis=0) for cls in self.classes])
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.prototypes = self.prototypes / np.where(norms > 1e-8, norms, 1)
        return self

    def predict(self, X):
        sims = cosine_similarity(X, self.prototypes)
        return np.array([self.classes[i] for i in sims.argmax(axis=1)])

def run_baseline_comparison(orig_feats, orig_labels):
    print("\n" + "="*55)
    print("实验一：纯原图多分类器对比（无增强）")
    print("="*55)

    ckpt = load_ckpt("baseline_comparison")
    if ckpt:
        print("  已完成，从断点恢复")
        return ckpt['results']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = {
        'LR (C=1)': LogisticRegression(C=1, max_iter=2000, class_weight='balanced', solver='lbfgs', multi_class='multinomial', random_state=42),
        'LR (C=5)': LogisticRegression(C=5, max_iter=2000, class_weight='balanced', solver='lbfgs', multi_class='multinomial', random_state=42),
        'Prototype': PrototypeClassifier(),
        'SVM-Linear': SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    }

    results = []
    for name, clf in classifiers.items():
        fold_f1 = []
        for train_idx, val_idx in skf.split(orig_feats, orig_labels):
            clf.fit(orig_feats[train_idx], orig_labels[train_idx])
            preds = clf.predict(orig_feats[val_idx])
            fold_f1.append(f1_score(orig_labels[val_idx], preds, average='macro'))
        mean_f1 = np.mean(fold_f1)
        std_f1  = np.std(fold_f1)
        results.append({'Classifier': name, 'Macro_F1_mean': mean_f1, 'Macro_F1_std': std_f1})
        print(f"  {name:<20}: {mean_f1:.4f} ± {std_f1:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'baseline_classifier_comparison.csv'), index=False)
    save_ckpt("baseline_comparison", {'results': results})
    print(f"  → 结果已保存到 {OUTPUT_DIR}/baseline_classifier_comparison.csv")
    return results

# ==================== 实验二：特征空间 MixUp ====================
def run_feature_space_mixup(orig_feats, orig_labels, orig_paths):
    print("\n" + "="*55)
    print("实验二：特征空间 MixUp（在特征向量上线性插值）")
    print("="*55)

    ckpt = load_ckpt("feature_space_mixup")
    if ckpt and 'final_f1' in ckpt:
        print(f"  已完成，从断点恢复: Macro F1 = {ckpt['final_f1']:.4f}")
        return ckpt['final_f1']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    fold_cache = {}

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(orig_feats, orig_labels)):
        fold_seed = RANDOM_SEED + fold_id * 100
        random.seed(fold_seed)
        np.random.seed(fold_seed)

        cls_to_indices = {cls: [] for cls in CLASS_NAMES}
        for i in train_idx:
            cls_to_indices[orig_labels[i]].append(i)

        mixup_feats, mixup_labels = [], []
        for idx in train_idx:
            cls = orig_labels[idx]
            feat_a = orig_feats[idx]
            candidates = [i for i in cls_to_indices[cls] if i != idx]
            if not candidates:
                candidates = cls_to_indices[cls]

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

        aug_feats  = np.array(mixup_feats)
        aug_labels = np.array(mixup_labels)
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")

    best_f1 = 0.0
    best_C  = svm_c_values[0]
    for C in svm_c_values:
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
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1 = mf1
            best_C = C

    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = SVC(kernel='rbf', C=best_C, gamma='scale', class_weight='balanced', random_state=42)
        clf.fit(tr_f, tr_l)
        y_pred_cv[cache['val_idx']] = clf.predict(orig_feats[cache['val_idx']])

    final_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    print(f"  最终 CV Macro F1 = {final_f1:.4f}")

    save_ckpt("feature_space_mixup", {'best_C': best_C, 'best_cv_f1': best_f1, 'final_f1': final_f1})
    return final_f1

# ==================== 实验三：图像空间 MixUp ====================
def run_image_space_mixup(orig_feats, orig_labels, orig_paths):
    print("\n" + "="*55)
    print("实验三：图像空间 MixUp（在图像上线性插值）")
    print("="*55)

    ckpt = load_ckpt("image_space_mixup")
    if ckpt and 'final_f1' in ckpt:
        print(f"  已完成，从断点恢复: Macro F1 = {ckpt['final_f1']:.4f}")
        return ckpt['final_f1']

    enhanced_aug = transforms.Compose([
        transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svm_c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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
            img_t = base_tf(img).unsqueeze(0).to(DEVICE)

            candidates = [p for p in cls_to_paths[cls] if p != img_path]
            if not candidates:
                candidates = cls_to_paths[cls]

            for _ in range(N_AUG):
                aug_tensor = enhanced_aug(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    f = model(aug_tensor)
                    f = f / f.norm(dim=-1, keepdim=True)
                std_feats.append(f.cpu().squeeze().numpy())
                std_labels.append(cls)

            for _ in range(N_AUG):
                other_path = random.choice(candidates)
                other_img  = Image.open(other_path).convert('RGB')
                other_t    = base_tf(other_img).unsqueeze(0).to(DEVICE)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mixed = lam * img_t + (1 - lam) * other_t
                with torch.no_grad():
                    f = model(mixed)
                    f = f / f.norm(dim=-1, keepdim=True)
                mixup_feats.append(f.cpu().squeeze().numpy())
                mixup_labels.append(cls)

        aug_feats  = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
        aug_labels = np.concatenate([np.array(std_labels), np.array(mixup_labels)])
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")

    best_f1 = 0.0
    best_C  = svm_c_values[0]
    for C in svm_c_values:
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
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1 = mf1
            best_C = C

    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = SVC(kernel='rbf', C=best_C, gamma='scale', class_weight='balanced', random_state=42)
        clf.fit(tr_f, tr_l)
        y_pred_cv[cache['val_idx']] = clf.predict(orig_feats[cache['val_idx']])

    final_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    print(f"  最终 CV Macro F1 = {final_f1:.4f}")

    save_ckpt("image_space_mixup", {'best_C': best_C, 'best_cv_f1': best_f1, 'final_f1': final_f1})
    return final_f1

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()

# 实验一：纯原图多分类器对比
results_baseline = run_baseline_comparison(orig_feats, orig_labels)

# 实验二：特征空间 MixUp
f1_feature_mixup = run_feature_space_mixup(orig_feats, orig_labels, orig_paths)

# 实验三：图像空间 MixUp
f1_image_mixup = run_image_space_mixup(orig_feats, orig_labels, orig_paths)

# ==================== 汇总输出 ====================
print("\n" + "="*60)
print("汇总：ConvNeXt-L 特征分类器对比 & MixUp空间对比")
print("="*60)

# 纯原图最优分类器
best_baseline = max(results_baseline, key=lambda x: x['Macro_F1_mean'])
print(f"纯原图最优分类器: {best_baseline['Classifier']} → F1 = {best_baseline['Macro_F1_mean']:.4f}")

# MixUp 空间对比
print(f"\nMixUp 空间对比:")
print(f"  特征空间 MixUp: F1 = {f1_feature_mixup:.4f}")
print(f"  图像空间 MixUp: F1 = {f1_image_mixup:.4f}")
print(f"  图像 vs 特征 差距: {f1_image_mixup - f1_feature_mixup:+.4f}")

# 保存最终对比
df_final = pd.DataFrame([
    {'Experiment': '纯原图最优分类器', 'Classifier': best_baseline['Classifier'], 'Macro_F1': best_baseline['Macro_F1_mean']},
    {'Experiment': '特征空间 MixUp', 'Classifier': 'RBF SVM', 'Macro_F1': f1_feature_mixup},
    {'Experiment': '图像空间 MixUp', 'Classifier': 'RBF SVM', 'Macro_F1': f1_image_mixup},
])
df_final.to_csv(os.path.join(OUTPUT_DIR, 'final_summary.csv'), index=False)
print(f"\n最终汇总已保存到 {OUTPUT_DIR}/final_summary.csv")
print("所有任务完成！")