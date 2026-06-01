import os
import torch
import open_clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
MODEL_NAME = 'ViT-L-14'
PRETRAINED = 'laion2b_s32b_b82k'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

# 单模板 Prompt
single_prompts = {
    "Class_0": "H&E stained histopathology, pleomorphic cell nuclei, hyperchromatic irregular nuclei, coarse chromatin, prominent nucleoli, densely packed, high nuclear-to-cytoplasm ratio, 40x magnification, digital pathology patch",
    "Class_1": "H&E stained histopathology, normal cell nuclei, regular round nuclei, uniform size, fine chromatin, smooth nuclear contours, loosely spaced, visible cytoplasm, 40x magnification, digital pathology patch",
    "Class_2": "H&E stained histopathology, moderately pleomorphic cell nuclei, dense nuclear clusters, coarse hyperchromatic chromatin, prominent nucleoli, overlapping nuclei, intermediate atypia, purple-violet staining, pink cytoplasmic background, 40x magnification, digital pathology patch",
    "Class_3": "H&E stained histopathology, spindle-shaped fusiform nuclei, cigar-like smooth nuclear contours, fine chromatin, fascicular arrangement, pale lavender staining, abundant eosinophilic stroma, very low cellularity, 40x magnification, digital pathology patch",
    "Class_4": "H&E stained histopathology, round to oval hyperchromatic nuclei, significant size variation, coarse granular chromatin, prominent nucleoli, dense packing, distinct nuclear membranes, poorly differentiated carcinoma, 40x magnification, digital pathology patch",
}

# 集成模板（每个类别三个变体）
ensemble_prompts = {
    "Class_0": [
        "H&E stained histopathology, pleomorphic cell nuclei, hyperchromatic irregular nuclei, coarse chromatin, prominent nucleoli, densely packed, 40x magnification",
        "H&E stained histopathology patch, densely packed pleomorphic nuclei, dark hyperchromatic staining, coarse granular chromatin, abnormal mitotic activity, 40x",
        "microscopy histopathology image, high nuclear pleomorphism, irregular hyperchromatic nuclei, nuclear crowding, abnormal cell growth, digital pathology",
    ],
    "Class_1": [
        "H&E stained histopathology, normal cell nuclei, regular round nuclei, uniform size, fine chromatin, loosely spaced, visible cytoplasm, 40x magnification",
        "H&E stained histopathology patch, benign cell nuclei, smooth nuclear contours, pale purple staining, abundant pink cytoplasm, well-differentiated, 40x",
        "microscopy histopathology image, normal nuclei morphology, low nuclear-to-cytoplasm ratio, fine evenly distributed chromatin, inconspicuous nucleoli, digital pathology",
    ],
    "Class_2": [
        "H&E stained histopathology, moderately pleomorphic nuclei, dense clusters, coarse chromatin, intermediate atypia, purple-violet staining, 40x magnification",
        "H&E stained histopathology patch, overlapping nuclei, moderate nuclear atypia, hyperchromatic coarse chromatin, prominent nucleoli in some cells, 40x",
        "microscopy histopathology image, intermediate dysplasia, cell-rich region, moderate nuclear pleomorphism, pink eosinophilic background, digital pathology",
    ],
    "Class_3": [
        "H&E stained histopathology, spindle-shaped fusiform nuclei, fascicular arrangement, pale lavender staining, abundant eosinophilic stroma, 40x magnification",
        "H&E stained histopathology patch, elongated cigar-shaped nuclei, wavy parallel arrangement, fine chromatin, very low cellularity, pale background, 40x",
        "microscopy histopathology image, mesenchymal spindle cells, smooth nuclear contours, fibrous stromal texture, light purple staining, digital pathology",
    ],
    "Class_4": [
        "H&E stained histopathology, round to oval hyperchromatic nuclei, significant size variation, coarse granular chromatin, poorly differentiated carcinoma, 40x magnification",
        "H&E stained histopathology patch, small round blue cell tumor, dense nuclear packing, dark hyperchromatic staining, distinct nuclear membranes, binucleated cells, 40x",
        "microscopy histopathology image, poorly differentiated nuclei, variable nuclear sizes, prominent nucleoli in some cells, pink cytoplasmic background, digital pathology",
    ],
}

# ==================== 加载模型 ====================
print(f"Loading model: {MODEL_NAME} ({PRETRAINED})...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(DEVICE).eval()

# ==================== 计算文本特征 ====================
def get_text_features(prompts_dict, ensemble):
    class_features = {}
    for cls in CLASS_NAMES:
        if ensemble:
            prompts = prompts_dict[cls]
            tokens = tokenizer(prompts).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            class_features[cls] = feats.mean(dim=0)
        else:
            prompt = prompts_dict[cls]
            tokens = tokenizer([prompt]).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            class_features[cls] = feat.squeeze(0)
    return class_features

print("\nComputing text features (Single)...")
text_feats_single = get_text_features(single_prompts, ensemble=False)
print("Computing text features (Ensemble)...")
text_feats_ens = get_text_features(ensemble_prompts, ensemble=True)

# ==================== 预测函数 ====================
def predict_image(img_path, class_feats):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    text_matrix = torch.stack([class_feats[c] for c in CLASS_NAMES])
    similarity = (img_feat @ text_matrix.T).squeeze(0)
    pred_idx = similarity.argmax().item()
    return CLASS_NAMES[pred_idx]

# ==================== 评估 ====================
def evaluate(class_feats, prompt_type):
    y_true, y_pred = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        images = sorted([f for f in os.listdir(class_dir) if f.endswith(('.png','.jpg','.jpeg'))])
        for img_name in tqdm(images, desc=f"Evaluating {class_name} ({prompt_type})"):
            img_path = os.path.join(class_dir, img_name)
            y_true.append(class_name)
            y_pred.append(predict_image(img_path, class_feats))
    return y_true, y_pred

print("\nEvaluating Single Prompt...")
y_true_s, y_pred_s = evaluate(text_feats_single, "Single")
print("\nEvaluating Ensemble Prompt...")
y_true_e, y_pred_e = evaluate(text_feats_ens, "Ensemble")

# ==================== 指标计算与保存 ====================
results = []
for name, y_true, y_pred in [("Single Prompt", y_true_s, y_pred_s),
                               ("Ensemble Prompt", y_true_e, y_pred_e)]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    results.append({
        'Prompt Strategy': name,
        'Accuracy': f"{acc*100:.2f}%",
        'Macro F1': f"{macro_f1:.4f}"
    })
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

# 保存到 CSV
df = pd.DataFrame(results)
df.to_csv('clip_zeroshot_results.csv', index=False)
print("\nResults saved to clip_zeroshot_results.csv")