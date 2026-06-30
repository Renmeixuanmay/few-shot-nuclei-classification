"""
DINOv2 ViT-S/14 + RBF SVM 多C集成交叉验证评估
对比：单C最优 vs 多C决策函数平均集成
自动保存所有对比数据至硬盘
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from torchvision import transforms
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
CACHE_DIR = "./dinov2_vits_ensemble_cache"
OUTPUT_DIR = "./dinov2_vits_ensemble_output"
CKPT_DIR   = "./dinov2_vits_ensemble_ckpt"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

INPUT_SIZE = 224
N_AUG = 30
C_CANDIDATES = [0.1, 0.5, 1.0, 5.0, 10.0]
N_RUNS = 3
RANDOM_SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

MIXUP_ALPHA = 0.4
GAUSSIAN_STD = 0.01
ERASING_P = 0.1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")
print(f"集成候选C: {C_CANDIDATES}")

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
print("加载 DINOv2 ViT-S/14...")
try:
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
except Exception as e:
    print(f"torch.hub 失败 ({e})，尝试 timm...")
    import timm
    model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
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

# ==================== 原图特征缓存 ====================
def extract_original_features():
    cache_feats = os.path.join(CACHE_DIR, "orig_feats.npy")
    cache_labels = os.path.join(CACHE_DIR, "orig_labels.npy")
    cache_paths = os.path.join(CACHE_DIR, "orig_paths.npy")
    if os.path.exists(cache_feats) and os.path.exists(cache_labels) and os.path.exists(cache_paths):
        print("✓ 加载原图特征缓存...")
        return (np.load(cache_feats), np.load(cache_labels), np.load(cache_paths, allow_pickle=True))
    print("提取原图特征 (250张)...")
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
    feats = np.array(feats)
    labels = np.array(labels)
    paths = np.array(paths)
    np.save(cache_feats, feats)
    np.save(cache_labels, labels)
    np.save(cache_paths, paths)
    print(f"  特征维度: {feats.shape}")
    return feats, labels, paths

# ==================== 无泄露增强特征生成 ====================
def generate_enhanced_features_for_fold(train_indices, orig_paths, orig_labels, n_aug, fold_seed):
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
        for _ in range(n_aug):
            aug_tensor = enhanced_aug(img).unsqueeze(0)
            feat = extract_feature(aug_tensor)
            std_feats.append(feat)
            std_labels.append(cls)
        for _ in range(n_aug):
            other_path = random.choice(candidates)
            other_img = Image.open(other_path).convert('RGB')
            other_t = base_tf(other_img).unsqueeze(0)
            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
            mixed = lam * img_t + (1 - lam) * other_t
            feat = extract_feature(mixed)
            mixup_feats.append(feat)
            mixup_labels.append(cls)
    all_aug_feats = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
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
            'val_idx': val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")
    return fold_cache

# ==================== 单次运行（集成评估 + 保存数据） ====================
def single_run(run_id, orig_feats, orig_labels, orig_paths):
    print(f"\n{'='*50}")
    print(f"Run {run_id}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_id*100)
    fold_cache = build_fold_cache(run_id, skf, orig_feats, orig_labels, orig_paths)

    fold_ensemble_f1 = []
    fold_best_single_f1 = []
    fold_best_C_list = []       # 记录每折的最佳C
    fold_records = []           # 保存每折详细数据

    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        val_f = orig_feats[cache['val_idx']]
        val_l = orig_labels[cache['val_idx']]

        svms = {}
        for C in C_CANDIDATES:
            clf = SVC(kernel='rbf', C=C, gamma='scale',
                      class_weight='balanced', random_state=42+run_id)
            clf.fit(tr_f, tr_l)
            svms[C] = clf

        # 单C最优
        single_f1s = {}
        for C, clf in svms.items():
            pred = clf.predict(val_f)
            single_f1s[C] = f1_score(val_l, pred, average='macro')
        best_C = max(single_f1s, key=single_f1s.get)
        best_single_f1 = single_f1s[best_C]
        fold_best_single_f1.append(best_single_f1)
        fold_best_C_list.append(best_C)

        # 多C集成
        decision_sum = None
        for C, clf in svms.items():
            dec = clf.decision_function(val_f)
            if decision_sum is None:
                decision_sum = dec
            else:
                decision_sum += dec
        avg_dec = decision_sum / len(svms)
        ensemble_pred = svms[C_CANDIDATES[0]].classes_[np.argmax(avg_dec, axis=1)]
        ensemble_f1 = f1_score(val_l, ensemble_pred, average='macro')
        fold_ensemble_f1.append(ensemble_f1)

        # 记录每折所有C的单C F1
        fold_record = {
            'Run': run_id,
            'Fold': fold_id,
            'Best_C': best_C,
            'Best_Single_F1': best_single_f1,
            'Ensemble_F1': ensemble_f1
        }
        for C in C_CANDIDATES:
            fold_record[f'Single_F1_C={C}'] = single_f1s.get(C, np.nan)
        fold_records.append(fold_record)

        print(f"  折 {fold_id}: 最佳单C={best_C}(F1={best_single_f1:.4f}), 集成F1={ensemble_f1:.4f}")

    mean_ensemble = np.mean(fold_ensemble_f1)
    mean_best_single = np.mean(fold_best_single_f1)
    print(f"\n  本跑平均: 集成={mean_ensemble:.4f}, 单C最优={mean_best_single:.4f}")

    # ----- 保存本次运行的每折数据 -----
    df_fold = pd.DataFrame(fold_records)
    df_fold.to_csv(os.path.join(OUTPUT_DIR, f'ensemble_fold_details_Run{run_id}.csv'), index=False)
    print(f"  → 每折详细数据已保存到 {OUTPUT_DIR}/ensemble_fold_details_Run{run_id}.csv")

    # 最终CV评估
    y_pred_ensemble = np.empty(len(orig_labels), dtype=object)
    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        val_f = orig_feats[cache['val_idx']]

        svms = {}
        for C in C_CANDIDATES:
            clf = SVC(kernel='rbf', C=C, gamma='scale',
                      class_weight='balanced', random_state=42+run_id)
            clf.fit(tr_f, tr_l)
            svms[C] = clf
        decision_sum = None
        for clf in svms.values():
            dec = clf.decision_function(val_f)
            decision_sum = dec if decision_sum is None else decision_sum + dec
        avg_dec = decision_sum / len(svms)
        pred = svms[C_CANDIDATES[0]].classes_[np.argmax(avg_dec, axis=1)]
        y_pred_ensemble[cache['val_idx']] = pred

    final_acc = accuracy_score(orig_labels, y_pred_ensemble)
    final_f1 = f1_score(orig_labels, y_pred_ensemble, average='macro')
    print(f"  最终CV集成: Acc={final_acc*100:.2f}%  Macro F1={final_f1:.4f}")

    save_ckpt(run_id, {
        'fold_ensemble_f1': fold_ensemble_f1,
        'fold_best_single_f1': fold_best_single_f1,
        'fold_best_C_list': fold_best_C_list,
        'final_acc': final_acc,
        'final_f1': final_f1,
        'y_pred': y_pred_ensemble.tolist()
    })

    # ----- 保存本次运行汇总 -----
    df_summary = pd.DataFrame([{
        'Run': run_id,
        'Mean_Best_Single_F1': mean_best_single,
        'Mean_Ensemble_F1': mean_ensemble,
        'Final_CV_Ensemble_Acc': final_acc,
        'Final_CV_Ensemble_F1': final_f1
    }])
    df_summary.to_csv(os.path.join(OUTPUT_DIR, f'ensemble_summary_Run{run_id}.csv'), index=False)
    print(f"  → 本次运行汇总已保存到 {OUTPUT_DIR}/ensemble_summary_Run{run_id}.csv")

    return mean_ensemble, mean_best_single, final_f1

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()

all_ensemble_f1, all_single_f1 = [], []
all_run_summaries = []

for run_id in range(1, N_RUNS+1):
    ckpt = load_ckpt(run_id)
    if ckpt and 'final_f1' in ckpt:
        print(f"\n>>>> Run {run_id} 已完成，恢复结果 <<<<")
        ens_f1 = ckpt['final_f1']
        mean_single = np.mean(ckpt['fold_best_single_f1'])
        all_ensemble_f1.append(ens_f1)
        all_single_f1.append(mean_single)
        all_run_summaries.append({
            'Run': run_id,
            'Mean_Best_Single_F1': mean_single,
            'Mean_Ensemble_F1': np.mean(ckpt['fold_ensemble_f1']),
            'Final_CV_Ensemble_Acc': ckpt.get('final_acc', np.nan),
            'Final_CV_Ensemble_F1': ens_f1
        })
        continue
    try:
        mean_ens, mean_single, final_f1 = single_run(run_id, orig_feats, orig_labels, orig_paths)
        all_ensemble_f1.append(final_f1)
        all_single_f1.append(mean_single)
        all_run_summaries.append({
            'Run': run_id,
            'Mean_Best_Single_F1': mean_single,
            'Mean_Ensemble_F1': mean_ens,
            'Final_CV_Ensemble_Acc': np.nan,
            'Final_CV_Ensemble_F1': final_f1
        })
    except Exception as e:
        print(f"Run {run_id} 失败: {e}")
        if interrupted:
            sys.exit(0)

# ==================== 保存最终汇总 ====================
print("\n" + "="*60)
print("多C集成 vs 单C最优 交叉验证对比")
print("="*60)
mean_all_ens = np.mean(all_ensemble_f1)
std_all_ens = np.std(all_ensemble_f1)
mean_all_single = np.mean(all_single_f1)
std_all_single = np.std(all_single_f1)
print(f"单C最优平均 F1: {mean_all_single:.4f} ± {std_all_single:.4f}")
print(f"多C集成平均 F1: {mean_all_ens:.4f} ± {std_all_ens:.4f}")
print(f"提升: {mean_all_ens - mean_all_single:.4f}")

# 保存最终对比汇总
df_final = pd.DataFrame([{
    'Method': 'DINOv2 ViT-S/14 + RBF SVM 多C集成',
    'C_Candidates': str(C_CANDIDATES),
    'Mean_Single_Best_F1': f"{mean_all_single:.4f}",
    'Std_Single_Best_F1': f"{std_all_single:.4f}",
    'Mean_Ensemble_F1': f"{mean_all_ens:.4f}",
    'Std_Ensemble_F1': f"{std_all_ens:.4f}",
    'Delta': f"{mean_all_ens - mean_all_single:.4f}",
    'N_Runs': N_RUNS
}])
df_final.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_final_comparison.csv'), index=False)
print(f"\n最终汇总已保存到 {OUTPUT_DIR}/ensemble_final_comparison.csv")

# 保存每次运行的汇总
df_all_runs = pd.DataFrame(all_run_summaries)
df_all_runs.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_all_runs_summary.csv'), index=False)
print(f"所有运行汇总已保存到 {OUTPUT_DIR}/ensemble_all_runs_summary.csv")