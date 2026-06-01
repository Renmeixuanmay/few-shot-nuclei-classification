import os
import torch
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, classification_report
import pandas as pd

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'laion2b_s32b_b82k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
K_VALUES = [1, 3, 5, 7]
PREFIX = "CLIP_KNN_"

# ==================== 加载 CLIP 视觉编码器 ====================
print(f"加载 CLIP 模型: {MODEL_NAME} ({PRETRAINED})...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()

# ==================== 特征提取（仅使用原图，无增强） ====================
def extract_features(train_dir):
    all_features, all_labels = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        images = sorted([f for f in os.listdir(class_dir)
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
        for img_name in tqdm(images, desc=f"提取 {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(img_tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)  # 余弦相似度 → 欧氏距离统一
            all_features.append(feat.cpu().squeeze().numpy())
            all_labels.append(class_name)
    return np.array(all_features), np.array(all_labels)

print("\n提取 CLIP ViT‑L 图像特征（仅原图，250 张）...")
features, labels = extract_features(TRAIN_DIR)
print(f"特征矩阵形状: {features.shape}")          # (250, 768)

# ==================== KNN 交叉验证 ====================
print("\nKNN Few‑Shot 交叉验证（5‑fold）...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
knn_results = []

for k in K_VALUES:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine')
    y_pred = cross_val_predict(knn, features, labels, cv=skf)
    acc = accuracy_score(labels, y_pred)
    macro_f1 = f1_score(labels, y_pred, average='macro')
    knn_results.append({'K': k, 'Accuracy': f"{acc*100:.2f}%", 'Macro F1': f"{macro_f1:.4f}"})
    print(f"  K={k}: Accuracy={acc*100:.2f}%, Macro F1={macro_f1:.4f}")

# 找出最佳 K
best_k_idx = np.argmax([f1_score(labels, cross_val_predict(KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine'), features, labels, cv=skf), average='macro') for k in K_VALUES])
best_k = K_VALUES[best_k_idx]
print(f"\n最佳 K = {best_k}")

# ==================== 最佳 K 的详细评估 ====================
knn_best = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='cosine')
y_pred_best = cross_val_predict(knn_best, features, labels, cv=skf)
print(f"\n===== KNN (K={best_k}) 分类报告 =====")
print(classification_report(labels, y_pred_best, digits=4))

# 保存结果
df_knn = pd.DataFrame(knn_results)
df_knn.to_csv(f'{PREFIX}knn_results.csv', index=False)
print(f"\n→ {PREFIX}knn_results.csv 已保存")

# 保存最佳 K 的完整评估
best_acc = accuracy_score(labels, y_pred_best)
best_f1  = f1_score(labels, y_pred_best, average='macro')
summary = [{'Method': f'CLIP ViT‑L + KNN (K={best_k})', 'Accuracy': f"{best_acc*100:.2f}%", 'Macro F1': f"{best_f1:.4f}"}]
pd.DataFrame(summary).to_csv(f'{PREFIX}best_knn_summary.csv', index=False)
print(f"→ {PREFIX}best_knn_summary.csv 已保存")
print("完成！")