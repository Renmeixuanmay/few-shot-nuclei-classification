"""
CLIP ViT-L/14 + Linear Probe (逻辑回归) 诚实评估
增强配置1：标准增强（翻转/旋转/颜色抖动，无MixUp），N_AUG=20
增强配置2：标准增强 + 图像空间 MixUp (α=0.2)，N_AUG=20
严格遵守无数据泄露原则，结果自动保存至硬盘
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
from sklearn.linear_model import LogisticRegression
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
CACHE_DIR  = "./clip_vitl_honest_cache"
OUTPUT_DIR = "./clip_vitl_honest_output"
CKPT_DIR   = "./clip_vitl_honest_ckpt"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

# ==================== 超参 ====================
INPUT_SIZE   = 224
N_AUG        = 20            # 与原始报告一致
MIXUP_ALPHA  = 0.2           # 原始报告中的 α
C_VALUES     = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
N_RUNS       = 3
RANDOM_SEED  = 42

# CLIP 官方归一化
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")
print("CLIP ViT-L/14 Linear Probe 诚实评估")

# ==================== 断点管理 ====================
def get_ckpt_path(config_name, run_id=None):
    if run_id is not None:
        return os.path.join(CKPT_DIR, f"ckpt_{config_name}_run{run_id}.json")
    return os.path.join(CKPT_DIR, f"ckpt_{config_name}.json")

def save_ckpt(config_name, data, run_id=None):
    with open(get_ckpt_path(config_name, run_id), 'w') as f:
        json.dump(data, f, indent=2)

def load_ckpt(config_name, run_id=None):
    path = get_ckpt_path(config_name, run_id)
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
print("加载 CLIP ViT-L/14...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='laion2b_s32b_b82k'
)
model = model.visual.to(DEVICE).eval()
# CLIP的视觉编码器不包含最后的 proj，我们直接用 image features
def extract_feature(img_tensor):
    with torch.no_grad():
        feat = model(img_tensor.to(DEVICE))
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

# 基础预处理（无增强）
base_tf = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# 标准增强（无MixUp）
standard_aug = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# 标准增强 + MixUp 用的增强管道（与标准增强相同，MixUp在特征生成时处理）
mixup_aug = transforms.Compose([
    transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ==================== 原始特征缓存 ====================
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

# ==================== 无泄露增强生成 ====================
def generate_enhanced_features_for_fold(train_indices, orig_paths, orig_labels,
                                        n_aug, fold_seed, use_mixup=False):
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    std_feats, mixup_feats = [], []
    std_labels, mixup_labels = [], []

    cls_to_paths = {cls: [] for cls in CLASS_NAMES}
    for i in train_indices:
        cls_to_paths[orig_labels[i]].append(orig_paths[i])

    aug_pipeline = mixup_aug if use_mixup else standard_aug

    for idx in tqdm(train_indices, desc="增强特征生成", leave=False):
        cls = orig_labels[idx]
        img_path = orig_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_t = base_tf(img).unsqueeze(0)

        candidates = [p for p in cls_to_paths[cls] if p != img_path]
        if not candidates:
            candidates = cls_to_paths[cls]

        # 标准增强
        for _ in range(n_aug):
            aug_tensor = aug_pipeline(img).unsqueeze(0)
            feat = extract_feature(aug_tensor)
            std_feats.append(feat)
            std_labels.append(cls)

        # MixUp（仅在启用时）
        if use_mixup:
            for _ in range(n_aug):
                other_path = random.choice(candidates)
                other_img  = Image.open(other_path).convert('RGB')
                other_t    = base_tf(other_img).unsqueeze(0)
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                mixed = lam * img_t + (1 - lam) * other_t
                feat = extract_feature(mixed)
                mixup_feats.append(feat)
                mixup_labels.append(cls)

    if mixup_feats:
        all_aug_feats  = np.concatenate([np.array(std_feats), np.array(mixup_feats)])
        all_aug_labels = np.concatenate([np.array(std_labels), np.array(mixup_labels)])
    else:
        all_aug_feats  = np.array(std_feats)
        all_aug_labels = np.array(std_labels)
    return all_aug_feats, all_aug_labels

# ==================== 单次运行 ====================
def single_run(run_id, config_name, use_mixup, orig_feats, orig_labels, orig_paths):
    print(f"\n{'='*50}")
    print(f"配置: {config_name} | Run {run_id}")
    print(f"{'='*50}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_id*100)
    fold_cache = {}
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(orig_feats, orig_labels)):
        fold_seed = RANDOM_SEED + fold_id*100 + run_id*10000
        aug_feats, aug_labels = generate_enhanced_features_for_fold(
            train_idx, orig_paths, orig_labels, N_AUG, fold_seed, use_mixup=use_mixup)
        fold_cache[fold_id] = {
            'train_idx': train_idx,
            'val_idx':   val_idx,
            'aug_feats': aug_feats,
            'aug_labels': aug_labels
        }
        print(f"  折 {fold_id}: 训练{len(train_idx)} → 增强后{len(train_idx)+len(aug_feats)}张, 验证{len(val_idx)}张")

    ckpt = load_ckpt(config_name, run_id)
    completed_C = ckpt.get('completed_C', []) if ckpt else []
    cv_records = ckpt.get('cv_records', []) if ckpt else []

    mean_f1s = []
    for C in C_VALUES:
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

            clf = LogisticRegression(
                C=C, max_iter=2000, solver='lbfgs',
                class_weight='balanced', multi_class='multinomial',
                random_state=42+run_id)
            clf.fit(tr_f, tr_l)
            pred = clf.predict(val_f)
            fold_f1.append(f1_score(val_l, pred, average='macro'))

        mf1 = np.mean(fold_f1)
        mean_f1s.append(mf1)
        completed_C.append(C)
        cv_records.append({'Run': run_id, 'C': C, 'Macro_F1': mf1})
        print(f"  C={C:.3f}  →  macro-F1={mf1:.4f}")
        save_ckpt(config_name, {'completed_C': completed_C, 'cv_records': cv_records}, run_id)

    best_C = C_VALUES[np.argmax(mean_f1s)]
    best_f1 = max(mean_f1s)
    print(f"  最优 C={best_C}, CV best macro-F1={best_f1:.4f}")

    # 最终评估
    y_pred_cv = np.empty(len(orig_labels), dtype=object)
    for fold_id, cache in fold_cache.items():
        tr_f = np.concatenate([orig_feats[cache['train_idx']], cache['aug_feats']])
        tr_l = np.concatenate([orig_labels[cache['train_idx']], cache['aug_labels']])
        clf = LogisticRegression(
            C=best_C, max_iter=2000, solver='lbfgs',
            class_weight='balanced', multi_class='multinomial',
            random_state=42+run_id)
        clf.fit(tr_f, tr_l)
        y_pred_cv[cache['val_idx']] = clf.predict(orig_feats[cache['val_idx']])

    acc = accuracy_score(orig_labels, y_pred_cv)
    macro_f1 = f1_score(orig_labels, y_pred_cv, average='macro')
    print(f"  Run {run_id}: Acc={acc*100:.2f}%  Macro F1={macro_f1:.4f}")

    save_ckpt(config_name, {
        'completed_C': completed_C,
        'cv_records': cv_records,
        'best_C': best_C,
        'best_cv_f1': best_f1,
        'final_acc': acc,
        'final_f1': macro_f1
    }, run_id)
    return best_C, macro_f1, acc

# ==================== 主流程 ====================
orig_feats, orig_labels, orig_paths = extract_original_features()

configs = [
    ('standard', False, '标准增强 (20x)'),
    ('mixup',   True,  '标准增强 + MixUp (20x)')
]

all_results = {}
for config_name, use_mixup, desc in configs:
    print(f"\n{'='*60}")
    print(f"开始评估: {desc}")
    print(f"{'='*60}")
    f1_list, acc_list = [], []
    for run_id in range(1, N_RUNS+1):
        ckpt = load_ckpt(config_name, run_id)
        if ckpt and 'final_f1' in ckpt:
            print(f"\n>>>> Run {run_id} 已完成，从断点恢复 <<<<")
            f1_list.append(ckpt['final_f1'])
            acc_list.append(ckpt['final_acc'])
            continue
        try:
            best_C, f1, acc = single_run(run_id, config_name, use_mixup,
                                         orig_feats, orig_labels, orig_paths)
            f1_list.append(f1)
            acc_list.append(acc)
        except Exception as e:
            print(f"Run {run_id} 失败: {e}")
            if interrupted:
                sys.exit(0)

    if f1_list:
        mean_f1 = np.mean(f1_list)
        std_f1  = np.std(f1_list)
        mean_acc = np.mean(acc_list)
        std_acc  = np.std(acc_list)
        all_results[desc] = {
            'Mean Acc': f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%",
            'Mean F1': f"{mean_f1:.4f} ± {std_f1:.4f}",
            'Best F1': f"{np.max(f1_list):.4f}"
        }
        print(f"\n{desc}: Acc={all_results[desc]['Mean Acc']}, "
              f"F1={all_results[desc]['Mean F1']}, Best={all_results[desc]['Best F1']}")

# 保存最终汇总
summary_df = pd.DataFrame([
    {
        'Method': desc,
        'Mean Accuracy': all_results[desc]['Mean Acc'],
        'Mean Macro F1': all_results[desc]['Mean F1'],
        'Best Macro F1': all_results[desc]['Best F1']
    }
    for _, _, desc in configs

])
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'clip_vitl_linear_probe_honest.csv'), index=False)
print(f"\n最终汇总已保存到 {OUTPUT_DIR}/clip_vitl_linear_probe_honest.csv")
print("\n所有任务完成！")