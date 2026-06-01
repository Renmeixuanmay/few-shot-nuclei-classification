import os
import torch
import timm
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
RETRIEVAL_DB_PATH = "retrieval_db.pkl"
LLM_MODEL_PATH = "/home/meixuan/models/deepseek-vl2-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
K = 5                    # 检索相似图片数量
N_SAMPLES = 20           # 验证样本数（每类抽 N_SAMPLES 张，总共 5×N_SAMPLES 张）
USE_LOCAL_LLM = True     # True=本地DeepSeek-VL2, False=最近邻回退

# ==================== 加载 DeepSeek-VL2（与 rag_llm_inference.py 完全一致）====================
print("加载 DeepSeek-VL2-Tiny 本地模型...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
import json

with open(f"{LLM_MODEL_PATH}/processor_config.json", "r") as f:
    config = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
vl_processor = DeepseekVLV2Processor(
    tokenizer=tokenizer,
    candidate_resolutions=config["candidate_resolutions"],
    patch_size=config["patch_size"],
    downsample_ratio=config["downsample_ratio"],
    image_mean=config.get("image_mean", [0.5, 0.5, 0.5]),
    image_std=config.get("image_std", [0.5, 0.5, 0.5]),
    normalize=config.get("normalize", True),
    mask_prompt=config.get("mask_prompt", False),
    ignore_id=config.get("ignore_id", -100),
)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
).cuda().eval()
print("DeepSeek-VL2-Tiny 加载完成")

# ==================== 加载检索库 ====================
with open(RETRIEVAL_DB_PATH, 'rb') as f:
    db = pickle.load(f)
train_features = db['features']
train_labels = db['labels']
train_paths = db['paths']

# ==================== 特征提取（用于检索，不需要 LLM） ====================
MODEL_NAME = 'eva02_large_patch14_448'
INPUT_SIZE = 448
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]

feature_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
feature_model = feature_model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def extract_feature(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_model(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

# ==================== 构建留一验证样本 ====================
print(f"\n准备留一验证样本（每类 {N_SAMPLES} 张）...")
val_samples = []
for cls_idx, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TRAIN_DIR, class_name)
    all_imgs = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))
    for img_name in sampled:
        val_samples.append({
            'path': os.path.join(class_dir, img_name),
            'true_label': class_name
        })

print(f"共 {len(val_samples)} 张验证样本")

# ==================== 留一验证推理 ====================
print(f"\n开始留一验证（每张图片需 2-3 秒）...")
y_true, y_pred = [], []

for sample in tqdm(val_samples, desc="RAG 验证中"):
    test_path = sample['path']
    true_label = sample['true_label']

    # 提取测试图片特征
    test_feat = extract_feature(test_path)

    # 检索相似图片（从全量训练集中检索，但排除自身）
    sims = cosine_similarity(test_feat.reshape(1, -1), train_features)[0]
    # 排除自身
    sims[train_paths == test_path] = -1
    top_indices = sims.argsort()[-K:][::-1]
    retrieved_paths = train_paths[top_indices]
    retrieved_labels = train_labels[top_indices]

    # 构建 Prompt
    prompt = f"你是一位病理学专家。以下是几张H&E染色的细胞核病理图像及其类别：\n\n"
    for i, (label, idx) in enumerate(zip(retrieved_labels, top_indices)):
        prompt += f"示例{i+1}：类别：{label}\n"
    prompt += f"\n现在请判断最后一张测试图片的类别。\n"
    prompt += f"类别选项：{', '.join(CLASS_NAMES)}\n"
    prompt += "请只回复类别名称，不要加任何解释。"

    # LLM 推理
    if USE_LOCAL_LLM:
        try:
            all_img_paths = retrieved_paths.tolist() + [test_path]
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{prompt}",
                    "images": all_img_paths,
                },
                {"role": "<|Assistant|>", "content": ""}
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = vl_processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(DEVICE)
            inputs_embeds = llm_model.prepare_inputs_embeds(**prepare_inputs)
            outputs = llm_model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=30, do_sample=False,
            )
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  LLM 推理失败 ({e})，回退到最近邻")
            answer = retrieved_labels[0]
    else:
        answer = retrieved_labels[0]

    # 匹配答案
    matched = False
    for cls in CLASS_NAMES:
        if cls in answer:
            y_pred.append(cls)
            matched = True
            break
    if not matched:
        y_pred.append(retrieved_labels[0])
    y_true.append(true_label)

# ==================== 评估 ====================
print(f"\n{'='*50}")
print(f"留一验证结果 (n={len(val_samples)})")
print(f"{'='*50}")
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {macro_f1:.4f}")
print(f"\n分类报告:")
print(classification_report(y_true, y_pred, digits=4))