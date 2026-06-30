"""
ConvNeXt-L + 多种分类器对比 (L1, L2, RBF SVM)
完全无数据泄露 + 复用现有特征缓存/增强缓存
增强输出：混淆矩阵数值CSV、分类报告、错误案例、置信度分布、t-SNE类内距离

"""

import os
import json
import torch
import timm
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
)
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
TRAIN_DIR  = "/home/meixuan/data/train_few_shot"
CACHE_DIR  = "./convnext_lanczos_cache"
OUTPUT_DIR = "./convnext_advanced_classifiers_output"
CKPT_DIR   = "./convnext_advanced_classifiers_ckpt"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 超参 ====================
CLASS_NAMES  = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
N_AUG        = 30
C_VALUES_L1  = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
C_VALUES_L2  = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
SVM_C_VALUES = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

N_RUNS       = 3
INPUT_SIZE   = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
MIXUP_ALPHA  = 0.4
GAUSSIAN_STD = 0.01
ERASING_P    = 0.1
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")

# ==================== 断点管理 ====================
def get_ckpt_path(run_id, classifier_name):
    return os.path.join(CKPT_DIR, f"ckpt_run{run_id}_{classifier_name}.json")

def save_ckpt(run_id, classifier_name, data):
    with open(get_ckpt_path(run_id, classifier_name), 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(run_id, classifier_name):
    path = get_ckpt_path(run_id, classifier_name)
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

# ==================== 模型加载 ====================
print("加载 ConvNeXt-L...")
model = timm.create_model('convnext_large', pretrained=True, num_classes=0)
model = model.to(DEVICE).eval()

base_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

enhanced_aug = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
    transforms.Lambda(lambda x: x + GAUSSIAN_STD * torch.randn_like(x)),
    transforms.RandomErasing(p=ERASING_P, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])

# ==================== 原始特征提取（复用缓存）====================
def extract_original_features():
    cache_feats  = os.path.join(CACHE_DIR, "orig_feats.npy")
    cache_labels = os.path.join(CACHE_DIR, "orig_labels.npy")
    cache_paths  = os.path.join(CACHE_DIR, "orig_paths.npy")
    if os.path.exists(cache_feats) and os.path.exists(cache_labels) and os.path.exists(cache_paths):
        print("✓ 加载本地特征缓存...")
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
        img_t = base_tf(img).unsqueeze(0).to(DEVICE)

        candidates = [p for p in cls_to_paths[cls] if p != img_path]
        if not candidates:
            candidates = cls_to_paths[cls]

        # 标准增强
        for _ in range(n_aug):
            aug_tensor = enhanced_aug(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model(aug_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            std_feats.append(feat.cpu().squeeze().numpy())
            std_labels.append(cls)

        # MixUp
        for _ in range(n_aug):
            other_path = random.choice(candidates)
            other_img  = Image.open(other_path).convert('RGB')
            other_t    = base_tf(other_img).unsqueeze(0).to(DEVICE)
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            mixed = lam * img_t + (1 - lam) * other_t
            with torch.no_grad():
                feat = model(mixed)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            mixup_feats.append(feat.cpu().squeeze().numpy())
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
    return fold_cache

# ==================== 单个分类器的一次运行（全面输出） ====================
def run_single_classifier(classifier_name, run_id, orig_feats, orig_labels, orig_paths, generate_plots=False):
    print(f"\n{'='*50}")
    print(f"分类器: {classifier_name} | Run {run_id}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_id*100)
    fold_cache = build_fold_cache(run_id, skf, orig_feats, orig_labels, orig_paths)

    if classifier_name == 'L1_LR':
        C_list = C_VALUES_L1
        def build_clf(C):
            return LogisticRegression(penalty='l1', C=C, solver='saga',
                                      max_iter=5000, class_weight='balanced',
                                      multi_class='multinomial',
                                      random_state=42+run_id)
    elif classifier_name == 'L2_LR':
        C_list = C_VALUES_L2
        def build_clf(C):
            return LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                      max_iter=2000, class_weight='balanced',
                                      multi_class='multinomial',
                                      random_state=42+run_id)
    elif classifier_name == 'RBF_SVM':
        C_list = SVM_C_VALUES
        def build_clf(C):
            return SVC(kernel='rbf', C=C, gamma='scale',
                       class_weight='balanced', random_state=42+run_id)
    else:
        raise ValueError("Unknown classifier")

    ckpt = load_ckpt(run_id, classifier_name)
    completed_C = ckpt.get('completed_C', []) if ckpt else []
    cv_records = ckpt.get('cv_records', []) if ckpt else []

    mean_f1s = []
    for C in C_list:
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

            clf = build_clf(C)
            clf.fit(tr_f, tr_l)
            pred = clf.predict(val_f)
            fold_f1.append(f1_score(val_l, pred, average='macro'))

        mf1 = np.mean(fold_f1)
        mean_f1s.append(mf1)
        completed_C.append(C)
        cv_records.append({'classifier': classifier_name, 'Run': run_id,
                           'C': C, 'Macro_F1': mf1})
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")
        save_ckpt(run_id, classifier_name, {
            'completed_C': completed_C,
            'cv_records': cv_records
        })

    best_C = C_list[np.argmax(mean_f1s)]
    best_f1 = max(mean_f1s)
    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    # 最终评估（同时收集决策值作为置信度）
    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    confidence = np.empty(len(orig_labels), dtype=float)
    all_decisions = np.zeros((len(orig_labels), len(CLASS_NAMES)))  # 新增，用于决策函数均值

    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = build_clf(best_C)
        clf.fit(tr_f, tr_l)
        val_f = orig_feats[cache['val_idx']]
        val_l = orig_labels[cache['val_idx']]
        pred = clf.predict(val_f)
        dec = clf.decision_function(val_f)           # (n_samples, n_classes)
        conf = np.max(dec, axis=1)                  # 取每一行的最大值作为置信度
        y_pred_cv[cache['val_idx']] = pred
        confidence[cache['val_idx']] = conf
        all_decisions[cache['val_idx']] = dec

    acc = accuracy_score(orig_labels, y_pred_cv)
    macro_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    print(f"  Run {run_id}: Acc={acc*100:.2f}%  Macro F1={macro_f1:.4f}")

    save_ckpt(run_id, classifier_name, {
        'completed_C': completed_C,
        'cv_records': cv_records,
        'best_C': best_C,
        'best_cv_f1': best_f1,
        'final_acc': acc,
        'final_f1': macro_f1,
        'y_pred': y_pred_cv.tolist()
    })

    # ===== 保存本次运行的关键数据到CSV（每次运行都执行）=====
    prefix = f"{classifier_name}_Run{run_id}"

    # C值搜索记录
    pd.DataFrame(cv_records).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_cv_f1.csv'), index=False)

    # 混淆矩阵
    cm = confusion_matrix(orig_labels, y_pred_cv, labels=CLASS_NAMES)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confusion_counts.csv'))
    pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confusion_normalized.csv'))

    # 分类报告
    p, r, f1_per, s = precision_recall_fscore_support(
        orig_labels, y_pred_cv, labels=CLASS_NAMES, average=None)
    rows = [{'Category': cls, 'Precision': p[i], 'Recall': r[i],
             'F1-score': f1_per[i], 'Support': s[i]} for i, cls in enumerate(CLASS_NAMES)]
    mp, mr, mf, _ = precision_recall_fscore_support(
        orig_labels, y_pred_cv, labels=CLASS_NAMES, average='macro')
    rows.append({'Category': 'Macro Avg', 'Precision': mp,
                 'Recall': mr, 'F1-score': mf, 'Support': len(orig_labels)})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_classification_report.csv'), index=False)

    # 错误案例（前10）
    errors = []
    for i in range(len(orig_labels)):
        if orig_labels[i] != y_pred_cv[i]:
            errors.append({'index': i, 'true': orig_labels[i], 'pred': y_pred_cv[i],
                           'confidence': confidence[i]})
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    pd.DataFrame(errors[:10]).to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_error_cases.csv'), index=False)

    # 置信度分布
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0, float('inf')]
    labels = ['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0', '>1.0']
    conf_dist = pd.cut(confidence, bins=bins, labels=labels, right=False)
    conf_stats = conf_dist.value_counts().sort_index()
    pd.DataFrame({'Confidence Range': conf_stats.index, 'Count': conf_stats.values}).to_csv(
        os.path.join(OUTPUT_DIR, f'{prefix}_confidence_distribution.csv'), index=False)

    # 各类别决策函数均值（新增）
    df_dec = pd.DataFrame(all_decisions, columns=CLASS_NAMES)
    df_dec['True'] = orig_labels
    dec_mean = df_dec.groupby('True')[CLASS_NAMES].mean()
    dec_mean.to_csv(os.path.join(OUTPUT_DIR, f'{prefix}_decision_mean.csv'))

    # t-SNE类内距离（仅当generate_plots时才进行t-SNE，因为比较耗时，但我们也保存一份CSV）
    if generate_plots:
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
        pd.DataFrame([intra_distances]).to_csv(
            os.path.join(OUTPUT_DIR, f'{prefix}_intra_class_distances.csv'), index=False)

    # ===== 图表输出（仅第 1 次运行生成图片） =====
    if generate_plots:
        # C值曲线
        plt.figure(figsize=(8, 5))
        plt.plot([str(c) for c in C_list], mean_f1s, marker='o', color='#2ecc71', linewidth=2, markersize=8)
        plt.xlabel('Regularization Strength C')
        plt.ylabel('Macro F1')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_cv_curve.png'), dpi=150)
        plt.close()

        # 混淆矩阵热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    cmap='Greens', vmin=0, vmax=1)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_confusion_matrix.png'), dpi=150)
        plt.close()

        # t-SNE可视化
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        feats_2d = tsne.fit_transform(orig_feats)
        plt.figure(figsize=(8, 6))
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        for i, cls in enumerate(CLASS_NAMES):
            mask = orig_labels == cls
            plt.scatter(feats_2d[mask, 0], feats_2d[mask, 1], c=colors[i], label=cls, alpha=0.7, s=60)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix}_tsne.png'), dpi=150)
        plt.close()

        # 终端打印
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

        print(f"  图表和表格已保存到 {OUTPUT_DIR}/")

    return best_f1, best_C, acc, macro_f1, cv_records

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()
print(f"特征维度: {orig_feats.shape}")

# ★ 在这里修改要运行的分类器列表
classifiers = ['RBF_SVM']   # 可改为 ['RBF_SVM'] 或 ['L1_LR'] 或 ['L2_LR']
summary = {}

for clf_name in classifiers:
    all_acc, all_f1 = [], []
    for run_id in range(1, N_RUNS+1):
        ckpt = load_ckpt(run_id, clf_name)
        if ckpt and 'final_f1' in ckpt:
            print(f"\n>>>> {clf_name} Run {run_id} 已完成，从断点恢复 <<<<")
            all_acc.append(ckpt['final_acc'])
            all_f1.append(ckpt['final_f1'])
            continue
        try:
            _, _, acc, f1, _ = run_single_classifier(
                clf_name, run_id, orig_feats, orig_labels, orig_paths,
                generate_plots=(run_id == 1))
            all_acc.append(acc)
            all_f1.append(f1)
        except Exception as e:
            print(f"  {clf_name} Run {run_id} 失败: {e}")
            if interrupted:
                sys.exit(0)
    if all_f1:
        summary[clf_name] = {
            'Mean Acc': f"{np.mean(all_acc)*100:.2f}% ± {np.std(all_acc)*100:.2f}%",
            'Mean F1': f"{np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}",
            'Best F1': f"{np.max(all_f1):.4f}"
        }

print("\n" + "="*60)
print("最终对比 (ConvNeXt-L + 增强流水线)")
print("="*60)
for name, metrics in summary.items():
    print(f"{name}: Acc={metrics['Mean Acc']}, F1={metrics['Mean F1']}, Best={metrics['Best F1']}")

# ==================== 绘制带误差棒的 C 值曲线（汇总三次运行） ====================
if len(summary) > 0:
    clf_name = classifiers[0]
    if clf_name == 'L1_LR':
        C_list = C_VALUES_L1
    elif clf_name == 'L2_LR':
        C_list = C_VALUES_L2
    else:
        C_list = SVM_C_VALUES

    # 收集三次运行每个 C 值的 F1
    all_runs_f1 = []
    for run_id in range(1, N_RUNS + 1):
        run_f1 = []
        for C in C_list:
            ckpt = load_ckpt(run_id, clf_name)
            if ckpt and 'cv_records' in ckpt:
                f1_val = [r['Macro_F1'] for r in ckpt['cv_records'] if r['C'] == C]
                run_f1.append(f1_val[0] if f1_val else np.nan)
            else:
                run_f1.append(np.nan)
        all_runs_f1.append(run_f1)

    # 转换为数组并计算均值、标准差
    f1_array = np.array(all_runs_f1)
    mean_f1 = np.nanmean(f1_array, axis=0)
    std_f1 = np.nanstd(f1_array, axis=0)

    # ===== 保存C值-F1数据到CSV =====
    run_ids = list(range(1, N_RUNS + 1))
    df_f1 = pd.DataFrame(all_runs_f1, index=[f"Run_{i}" for i in run_ids], columns=[str(c) for c in C_list])
    df_f1.loc['Mean'] = mean_f1
    df_f1.loc['Std']  = std_f1
    csv_f1_path = os.path.join(OUTPUT_DIR, f'{clf_name}_c_f1_all_runs.csv')
    df_f1.to_csv(csv_f1_path)
    print(f"✓ 所有运行的C值-F1数据已保存到 {csv_f1_path}")

    # ===== 保存汇总结果到CSV =====
    summary_df = pd.DataFrame([{
        'Classifier': clf_name,
        'Mean Accuracy (%)': f"{np.mean(all_acc)*100:.2f}",
        'Std Accuracy (%)': f"{np.std(all_acc)*100:.2f}",
        'Mean Macro F1': f"{np.mean(all_f1):.4f}",
        'Std Macro F1': f"{np.std(all_f1):.4f}",
        'Best Macro F1': f"{max(all_f1):.4f}",
        'N_Runs': N_RUNS
    }])
    csv_sum_path = os.path.join(OUTPUT_DIR, f'{clf_name}_summary.csv')
    summary_df.to_csv(csv_sum_path, index=False)
    print(f"✓ 汇总结果已保存到 {csv_sum_path}")

    # 绘制带误差棒的曲线
    plt.figure(figsize=(8, 5))
    plt.errorbar(range(len(C_list)), mean_f1, yerr=std_f1,
                 marker='o', color='#2ecc71', linewidth=2, markersize=8,
                 capsize=5, capthick=1.5, elinewidth=1.5)
    plt.xticks(range(len(C_list)), [str(c) for c in C_list])
    plt.xlabel('Regularization Strength C')
    plt.ylabel('Macro F1')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{clf_name}_cv_curve_errorbar.png'), dpi=150)
    plt.close()
    print(f"带误差棒的 C 值曲线已保存到 {OUTPUT_DIR}/{clf_name}_cv_curve_errorbar.png")