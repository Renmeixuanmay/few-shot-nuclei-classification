"""
CONCH + RBF SVM (无数据泄露)
增强策略：标准增强 + 图像空间 MixUp (α=0.4) + 高斯噪声 (0.01) + 随机擦除 (p=0.1)
分类器：RBF SVM (细粒度 C 搜索)
全面输出：混淆矩阵、分类报告、错误案例、置信度分布、
          t‑SNE 类内距离、决策函数均值、C 值搜索记录等

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
from sklearn.manifold import TSNE
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
TRAIN_DIR       = "/home/meixuan/data/train_few_shot"
CACHE_DIR       = "./conch_svm_cache"
OUTPUT_DIR      = "./conch_svm_output"
CKPT_DIR        = "./conch_svm_ckpt"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# 加载 CONCH 模型
sys.path.insert(0, "/home/meixuan/CONCH")
from conch.open_clip_custom import create_model_from_pretrained

MODEL_PATH      = "/home/meixuan/models/CONCH"
CHECKPOINT_FILE = f"{MODEL_PATH}/pytorch_model.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

# ==================== 超参 ====================
INPUT_SIZE   = 224
N_AUG        = 30
SVM_C_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
N_RUNS       = 3
RANDOM_SEED  = 42

# CONCH 使用 CLIP 归一化（因为基于 CLIP）
MEAN = [0.48145466, 0.4578275,  0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

MIXUP_ALPHA  = 0.4
GAUSSIAN_STD = 0.01
ERASING_P    = 0.1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")
print("CONCH + 图像空间 MixUp + 高斯噪声 + 随机擦除 + RBF SVM")

# ==================== 断点管理 ====================
def get_ckpt_path(run_id):
    return os.path.join(CKPT_DIR, f"ckpt_run{run_id}.json")

def save_ckpt(run_id, data):
    with open(get_ckpt_path(run_id), 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(run_id):
    path = get_ckpt_path(run_id)
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
print("加载 CONCH 模型...")
model, preprocess = create_model_from_pretrained(
    'conch_ViT-B-16',
    checkpoint_path=CHECKPOINT_FILE,
    force_image_size=224
)
model = model.to(DEVICE).eval()

# 基础预处理（用于原图和 MixUp 混合前的变换）
base_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# 增强预处理（含噪声、擦除）
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
    feat = model.encode_image(img_tensor.to(DEVICE))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

# ==================== 原始图像特征缓存 ====================
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

# ==================== 无泄露增强特征生成 ====================
def generate_enhanced_features_for_fold(train_indices, orig_paths, orig_labels,
                                        n_aug, fold_seed):
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    std_feats, mixup_feats = [], []
    std_labels, mixup_labels = [], []

    cls_to_paths = {cls: [] for cls in CLASS_NAMES}
    for i in train_indices:
        cls_to_paths[orig_labels[i]].append(orig_paths[i])

    for idx in tqdm(train_indices, desc="增强特征生成", leave=False):
        cls = orig_labels[idx]
        img_path = orig_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_t = base_tf(img).unsqueeze(0)

        candidates = [p for p in cls_to_paths[cls] if p != img_path]
        if not candidates:
            candidates = cls_to_paths[cls]

        # 标准增强（含噪声/擦除）
        for _ in range(n_aug):
            aug_tensor = enhanced_aug(img).unsqueeze(0)
            feat = extract_feature(aug_tensor)
            std_feats.append(feat)
            std_labels.append(cls)

        # MixUp
        for _ in range(n_aug):
            other_path = random.choice(candidates)
            other_img  = Image.open(other_path).convert('RGB')
            other_t    = base_tf(other_img).unsqueeze(0)
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            mixed = lam * img_t + (1 - lam) * other_t
            feat = extract_feature(mixed)
            mixup_feats.append(feat)
            mixup_labels.append(cls)

    all_aug_feats  = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
    all_aug_labels = np.concatenate([np.array(std_labels), np.array(mixup_labels)])
    return all_aug_feats, all_aug_labels

def build_fold_cache(run_id, skf, orig_feats, orig_labels, orig_paths):
    fold_cache = {}
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(orig_feats, orig_labels)):
        fold_seed = RANDOM_SEED + fold_id*100 + run_id*10000
        aug_feats, aug_labels = generate_enhanced_features_for_fold(
            train_idx, orig_paths, orig_labels, N_AUG, fold_seed)
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx':   val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")
    return fold_cache

# ==================== 单次运行（全面输出） ====================
def single_run(run_id, orig_feats, orig_labels, orig_paths, generate_plots=False):
    print(f"\n{'='*50}")
    print(f"Run {run_id}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_id*100)
    fold_cache = build_fold_cache(run_id, skf, orig_feats, orig_labels, orig_paths)

    ckpt = load_ckpt(run_id)
    completed_C = ckpt.get('completed_C', []) if ckpt else []
    cv_records = ckpt.get('cv_records', []) if ckpt else []

    mean_f1s = []
    for C in SVM_C_VALUES:
        if C in completed_C:
            mf1 = [r['Macro_F1'] for r in cv_records if r['C']==C][0]
            mean_f1s.append(mf1)
            print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}  [恢复]")
            continue

        fold_f1 = []
        for fold_id, cache in fold_cache.items():
            tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
            tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
            val_f = orig_feats[cache['val_idx']]
            val_l = orig_labels[cache['val_idx']]

            clf = SVC(kernel='rbf', C=C, gamma='scale',
                      class_weight='balanced', random_state=42+run_id)
            clf.fit(tr_f, tr_l)
            pred = clf.predict(val_f)
            fold_f1.append(f1_score(val_l, pred, average='macro'))

        mf1 = np.mean(fold_f1)
        mean_f1s.append(mf1)
        completed_C.append(C)
        cv_records.append({'Run': run_id, 'C': C, 'Macro_F1': mf1})
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")
        save_ckpt(run_id, {'completed_C': completed_C, 'cv_records': cv_records})

    best_C = SVM_C_VALUES[np.argmax(mean_f1s)]
    best_f1 = max(mean_f1s)
    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    # 最终评估
    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    confidence = np.empty(len(orig_labels), dtype=float)
    all_decisions = np.zeros((len(orig_labels), len(CLASS_NAMES)))

    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = SVC(kernel='rbf', C=best_C, gamma='scale',
                  class_weight='balanced', random_state=42+run_id)
        clf.fit(tr_f, tr_l)
        val_f = orig_feats[cache['val_idx']]
        pred = clf.predict(val_f)
        dec = clf.decision_function(val_f)
        conf = np.max(dec, axis=1)
        y_pred_cv[cache['val_idx']] = pred
        confidence[cache['val_idx']] = conf
        all_decisions[cache['val_idx']] = dec

    acc = accuracy_score(orig_labels, y_pred_cv)
    macro_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    print(f"  Run {run_id}: Acc={acc*100:.2f}%  Macro F1={macro_f1:.4f}")

    save_ckpt(run_id, {
        'completed_C': completed_C,
        'cv_records': cv_records,
        'best_C': best_C,
        'best_cv_f1': best_f1,
        'final_acc': acc,
        'final_f1': macro_f1,
        'y_pred': y_pred_cv.tolist(),
        'mean_f1s': mean_f1s
    })

    # ===== 图表与详细表格输出（仅第 1 次运行） =====
    if generate_plots:
        prefix = f"CONCH_SVM_Run{run_id}"

        # --- C 值曲线 ---
        plt.figure(figsize=(8, 5))
        plt.plot([str(c) for c in SVM_C_VALUES], mean_f1s, marker='o', color='#9b59b6', linewidth=2, markersize=8)
        plt.xlabel('Regularization Strength C')
        plt.ylabel('Macro F1')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_C_curve.png'), dpi=150)
        plt.close()

        # --- 混淆矩阵 ---
        cm = confusion_matrix(orig_labels, y_pred_cv, labels=CLASS_NAMES)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(OUTPUT_DIR, f'{prefix}_confusion_counts.csv'))
        pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            os.path.join(OUTPUT_DIR, f'{prefix}_confusion_normalized.csv'))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cmap='Purples', vmin=0, vmax=1)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_confusion_matrix.png'), dpi=150)
        plt.close()

        print("\n归一化混淆矩阵:")
        print("           " + "  ".join(f"{c:>8}" for c in CLASS_NAMES))
        for i, cls in enumerate(CLASS_NAMES):
            row_str = "  ".join(f"{cm_norm[i, j]:.4f}" for j in range(len(CLASS_NAMES)))
            print(f"{cls:>8}  {row_str}")

        # --- 分类报告 ---
        p, r, f1_per, s = precision_recall_fscore_support(
            orig_labels, y_pred_cv, labels=CLASS_NAMES, average=None)
        rows = [{'Category': cls, 'Precision': p[i], 'Recall': r[i],
                 'F1-score': f1_per[i], 'Support': s[i]} for i, cls in enumerate(CLASS_NAMES)]
        mp, mr, mf, _ = precision_recall_fscore_support(
            orig_labels, y_pred_cv, labels=CLASS_NAMES, average='macro')
        rows.append({'Category': 'Macro Avg', 'Precision': mp,
                     'Recall': mr, 'F1-score': mf, 'Support': len(orig_labels)})
        df_report = pd.DataFrame(rows)
        df_report.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_classification_report.csv'), index=False)

        print(f"\n{'Category':<12} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        print("-" * 55)
        for row in rows:
            print(f"{row['Category']:<12} {row['Precision']:>10.4f} {row['Recall']:>10.4f} "
                  f"{row['F1-score']:>10.4f} {row['Support']:>10}")

        # --- 错误案例分析 ---
        errors = []
        for i in range(len(orig_labels)):
            if orig_labels[i] != y_pred_cv[i]:
                errors.append({'index': i, 'true': orig_labels[i], 'pred': y_pred_cv[i],
                               'confidence': confidence[i]})
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        pd.DataFrame(errors[:10]).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_error_cases.csv'), index=False)

        print(f"\n{'#':<4} {'True':<12} {'Predicted':<12} {'Confidence':>10}")
        print("-" * 42)
        for i, err in enumerate(errors[:5]):
            print(f"{i+1:<4} {err['true']:<12} {err['pred']:<12} {err['confidence']:>10.4f}")

        # --- 置信度分布 ---
        bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0, float('inf')]
        labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0', '>1.0']
        conf_dist = pd.cut(confidence, bins=bins, labels=labels, right=False)
        conf_stats = conf_dist.value_counts().sort_index()
        pd.DataFrame({'Confidence Range': conf_stats.index, 'Count': conf_stats.values}).to_csv(
            os.path.join(OUTPUT_DIR, f'{prefix}_confidence_distribution.csv'), index=False)

        # --- 各类别决策函数均值 ---
        df_dec = pd.DataFrame(all_decisions, columns=CLASS_NAMES)
        df_dec['True'] = orig_labels
        dec_mean = df_dec.groupby('True')[CLASS_NAMES].mean()
        dec_mean.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_decision_mean.csv'))
        print("\n各类别平均决策函数:")
        print(dec_mean.round(4))

        # --- t-SNE 可视化与类内距离 ---
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        feats_2d = tsne.fit_transform(orig_feats)

        intra_distances = {}
        for cls in CLASS_NAMES:
            mask = orig_labels == cls
            cls_feats = feats_2d[mask]
            if len(cls_feats) > 1:
                dist_sum = 0.0
                count = 0
                for i in range(len(cls_feats)):
                    for j in range(i+1, len(cls_feats)):
                        dist_sum += np.linalg.norm(cls_feats[i] - cls_feats[j])
                        count += 1
                intra_distances[cls] = dist_sum / count
            else:
                intra_distances[cls] = 0.0

        print("\nt‑SNE 二维空间各类别类内平均距离:")
        for cls in CLASS_NAMES:
            print(f"  {cls}: {intra_distances[cls]:.4f}")

        pd.DataFrame([intra_distances]).to_csv(
            os.path.join(OUTPUT_DIR, f'{prefix}_intra_class_distances.csv'), index=False)

        plt.figure(figsize=(8, 6))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for i, cls in enumerate(CLASS_NAMES):
            mask = orig_labels == cls
            plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=colors[i], label=cls, alpha=0.7, s=60)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_tsne.png'), dpi=150)
        plt.close()

        # --- C 值搜索记录 ---
        pd.DataFrame(cv_records).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_cv_f1.csv'), index=False)

        print(f"\n  图表和 CSV 已保存到 {OUTPUT_DIR}/")

    return best_f1, best_C, acc, macro_f1, mean_f1s

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()

all_accs, all_f1s = [], []
all_mean_f1s = []
best_overall = 0.0

for run_id in range(1, N_RUNS+1):
    ckpt = load_ckpt(run_id)
    if ckpt and 'final_f1' in ckpt:
        print(f"\n>>>> Run {run_id} 已完成，恢复结果 <<<<")
        all_accs.append(ckpt['final_acc'])
        all_f1s.append(ckpt['final_f1'])
        if 'mean_f1s' in ckpt:
            all_mean_f1s.append(ckpt['mean_f1s'])
        if ckpt['final_f1'] > best_overall:
            best_overall = ckpt['final_f1']
        continue
    try:
        _, _, acc, f1, mean_f1s = single_run(run_id, orig_feats, orig_labels, orig_paths,
                                             generate_plots=(run_id==1))
        all_accs.append(acc)
        all_f1s.append(f1)
        all_mean_f1s.append(mean_f1s)
        if f1 > best_overall:
            best_overall = f1
    except Exception as e:
        print(f"Run {run_id} 失败: {e}")
        if interrupted:
            sys.exit(0)

print("\n" + "="*60)
print("汇总: CONCH + RBF SVM")
print("="*60)
mean_acc = np.mean(all_accs); std_acc = np.std(all_accs)
mean_f1  = np.mean(all_f1s);  std_f1  = np.std(all_f1s)
print(f"Accuracy : {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}")
for i, (a, f) in enumerate(zip(all_accs, all_f1s)):
    print(f"  Run {i+1}: {a*100:.2f}%  F1={f:.4f}")
print(f"最佳单次 F1: {best_overall:.4f}")

summary_df = pd.DataFrame([{
    'Method': 'CONCH + RBF SVM',
    'Mean Accuracy (%)': f"{mean_acc*100:.2f}",
    'Std Accuracy (%)': f"{std_acc*100:.2f}",
    'Mean Macro F1': f"{mean_f1:.4f}",
    'Std Macro F1': f"{std_f1:.4f}",
    'Best Macro F1': f"{best_overall:.4f}",
    'N_Runs': N_RUNS
}])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'CONCH_SVM_summary.csv'), index=False)
print(f"汇总结果已保存到 {OUTPUT_DIR}/")

# ==================== 绘制带误差棒的 C 值曲线 ====================
if len(all_mean_f1s) == N_RUNS:
    f1_array = np.array(all_mean_f1s)
    mean_per_c = np.mean(f1_array, axis=0)
    std_per_c = np.std(f1_array, axis=0)

    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(SVM_C_VALUES)), mean_per_c, yerr=std_per_c,
                 marker='o', color='#9b59b6', linewidth=2, markersize=8,
                 capsize=5, capthick=1.5, elinewidth=1.5)
    plt.xticks(range(len(SVM_C_VALUES)), [str(c) for c in SVM_C_VALUES])
    plt.xlabel('Regularization Strength C')
    plt.ylabel('Macro F1')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'CONCH_SVM_C_curve_errorbar.png'), dpi=150)
    plt.close()
    print(f"带误差棒的 C 值曲线已保存到 {OUTPUT_DIR}/")
else:
    print("警告：未收集到完整的3次运行数据，无法绘制误差棒曲线。")