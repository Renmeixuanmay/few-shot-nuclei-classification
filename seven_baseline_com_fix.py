"""
七种预训练模型纯原图基线一键对比
模型列表：DINOv2 ViT-G, EVA02-L/14, CLIP ViT-L, ResNet50, ConvNeXt-XL, ConvNeXt-L, CONCH
评估方式：纯原图特征 + 逻辑回归 (C值搜索) + 5折交叉验证
完全无数据泄露，结果保存至硬盘
"""

import os
import json
import torch
import timm
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import random
import signal
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
BASE_CACHE_DIR = "./baseline_models_cache"
OUTPUT_DIR = "./baseline_models_output"
os.makedirs(BASE_CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
N_RUNS = 3
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(f"设备: {DEVICE}")
print(f"C值搜索范围: {C_VALUES}")
print(f"独立运行次数: {N_RUNS}")

# ==================== 断点管理 ====================
CKPT_DIR = "./baseline_models_ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
def get_ckpt_path(model_name):
    safe_name = model_name.replace('/', '_')
    return os.path.join(CKPT_DIR, f"ckpt_{safe_name}.json")

def save_ckpt(model_name, data):
    with open(get_ckpt_path(model_name), 'w') as f:
        json.dump(data, f, indent=2)
def load_ckpt(model_name):
    path = get_ckpt_path(model_name)
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

# ==================== 模型配置 ====================
MODELS_CONFIG = {
    "DINOv2_ViT-G": {
        "type": "dinov2",
        "model_name": "dinov2_vitg14",
        "input_size": 518,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "BICUBIC"
    },
    "EVA02-L/14": {
        "type": "timm",
        "model_name": "eva02_large_patch14_448",
        "input_size": 448,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "interpolation": "BICUBIC"
    },
    "CLIP_ViT-L": {
        "type": "open_clip",
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
        "input_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "interpolation": "BICUBIC"
    },
    "ResNet50": {
        "type": "timm",
        "model_name": "resnet50",
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "BICUBIC"
    },
    "ConvNeXt-XL": {
        "type": "timm",
        "model_name": "convnext_xlarge",
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "BICUBIC"
    },
    "ConvNeXt-L": {
        "type": "timm",
        "model_name": "convnext_large",
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "BICUBIC"
    },
    "CONCH": {
        "type": "conch",
        "model_name": "conch_ViT-B-16",
        "input_size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
        "interpolation": "BICUBIC"
    }
}

# ==================== 特征提取函数 ====================
def load_or_extract_features(model_name, config):
    cache_dir = os.path.join(BASE_CACHE_DIR, model_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_feats  = os.path.join(cache_dir, "orig_feats.npy")
    cache_labels = os.path.join(cache_dir, "orig_labels.npy")
    cache_paths  = os.path.join(cache_dir, "orig_paths.npy")

    if os.path.exists(cache_feats) and os.path.exists(cache_labels) and os.path.exists(cache_paths):
        print(f"  ✓ 加载 {model_name} 原图特征缓存...")
        return (np.load(cache_feats), np.load(cache_labels), np.load(cache_paths, allow_pickle=True))

    # 加载模型
    print(f"  加载 {model_name} 模型...")
    if config["type"] == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', config["model_name"])
    elif config["type"] == "timm":
        model = timm.create_model(config["model_name"], pretrained=True, num_classes=0)
    elif config["type"] == "open_clip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            config["model_name"], pretrained=config["pretrained"])
        model = model.visual
    elif config["type"] == "conch":
        sys.path.insert(0, "/home/meixuan/CONCH")
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained(
            config["model_name"],
            checkpoint_path="/home/meixuan/models/CONCH/pytorch_model.bin",
            force_image_size=224)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
    model = model.to(DEVICE).eval()

    # 预处理
    from torchvision import transforms
    interpolation = transforms.InterpolationMode.BICUBIC
    preprocess = transforms.Compose([
        transforms.Resize(config["input_size"], interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["mean"], std=config["std"])
    ])

    print(f"  提取 {model_name} 原图特征（250张）...")
    feats, labels, paths = [], [], []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        imgs = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        for fn in tqdm(imgs, desc=f"    {cls}"):
            img_path = os.path.join(cls_dir, fn)
            img = Image.open(img_path).convert('RGB')
            t = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if config["type"] == "conch":
                    f = model.encode_image(t)  # CONCH专用视觉编码方法
                elif config["type"] in ["dinov2", "open_clip"]:
                    f = model(t)
                
                else:  # timm models output [B, D]
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
    print(f"    特征维度: {feats.shape}")
    # 释放模型
    del model
    torch.cuda.empty_cache()
    return feats, labels, paths

# ==================== 单模型单次运行 ====================
def single_run(model_name, run_id, feats, labels):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + run_id*100)
    best_f1 = 0.0
    best_C = C_VALUES[0]
    for C in C_VALUES:
        fold_f1 = []
        for train_idx, val_idx in skf.split(feats, labels):
            clf = LogisticRegression(C=C, max_iter=2000, solver='lbfgs', class_weight='balanced',
                                     multi_class='multinomial', random_state=42+run_id)
            clf.fit(feats[train_idx], labels[train_idx])
            pred = clf.predict(feats[val_idx])
            fold_f1.append(f1_score(labels[val_idx], pred, average='macro'))
        mf1 = np.mean(fold_f1)
        if mf1 > best_f1:
            best_f1 = mf1
            best_C = C
    # 最终评估用最优C
    y_pred = np.empty(len(labels), dtype=object)
    for train_idx, val_idx in skf.split(feats, labels):
        clf = LogisticRegression(C=best_C, max_iter=2000, solver='lbfgs', class_weight='balanced',
                                 multi_class='multinomial', random_state=42+run_id)
        clf.fit(feats[train_idx], labels[train_idx])
        y_pred[val_idx] = clf.predict(feats[val_idx])
    acc = accuracy_score(labels, y_pred)
    macro_f1 = f1_score(labels, y_pred, average='macro')
    return best_C, macro_f1, acc

# ==================== 主流程 ====================
all_results = []

for model_name, config in MODELS_CONFIG.items():
    print(f"\n{'='*60}")
    print(f"处理模型: {model_name}")
    print(f"{'='*60}")
    ckpt = load_ckpt(model_name)
    if ckpt and 'final_f1_mean' in ckpt:
        print(f"  已完成，从断点恢复")
        all_results.append({
            '模型': model_name,
            '预训练范式': config.get('paradigm', ''),
            '参数量': config.get('params', ''),
            'Macro F1 均值': ckpt['final_f1_mean'],
            'Macro F1 标准差': ckpt['final_f1_std'],
            '最佳单次': ckpt['best_single'],
            '最优C': ckpt['best_C_overall']
        })
        continue

    try:
        feats, labels, _ = load_or_extract_features(model_name, config)
    except Exception as e:
        print(f"  ✗ 特征提取失败: {e}")
        continue

    f1_list = []
    acc_list = []
    best_C_list = []
    for run_id in range(1, N_RUNS+1):
        best_C, f1, acc = single_run(model_name, run_id, feats, labels)
        f1_list.append(f1)
        acc_list.append(acc)
        best_C_list.append(best_C)
        print(f"  Run {run_id}: C={best_C:.4f}, Acc={acc*100:.2f}%, Macro F1={f1:.4f}")

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    best_single = np.max(f1_list)
    # 选择出现次数最多的C作为总体最优
    from collections import Counter
    best_C_overall = Counter(best_C_list).most_common(1)[0][0]

    print(f"  {model_name}: F1 = {mean_f1:.4f} ± {std_f1:.4f}, 最佳单次 = {best_single:.4f}")
    all_results.append({
        '模型': model_name,
        '预训练范式': '',
        '参数量': '',
        'Macro F1 均值': f"{mean_f1:.4f}",
        'Macro F1 标准差': f"{std_f1:.4f}",
        '最佳单次': f"{best_single:.4f}",
        '最优C': best_C_overall
    })
    save_ckpt(model_name, {
        'final_f1_mean': mean_f1,
        'final_f1_std': std_f1,
        'best_single': best_single,
        'best_C_overall': best_C_overall
    })

# ==================== 汇总输出 ====================
print("\n" + "="*80)
print("七种预训练模型纯原图基线对比（逻辑回归，无增强）")
print("="*80)
df = pd.DataFrame(all_results)
df.to_csv(os.path.join(OUTPUT_DIR, "pure_baseline_comparison.csv"), index=False)
print(df.to_string(index=False))
print(f"\n结果已保存到 {OUTPUT_DIR}/pure_baseline_comparison.csv")