import os
import torch
import timm
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
import io
import warnings
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
TEST_DIR = "/home/meixuan/data/test"                    # 测试集路径（老师发布后修改）
RETRIEVAL_DB_PATH = "retrieval_db.pkl"                  # 检索库路径
MODEL_NAME = 'eva02_large_patch14_448'                  # 特征提取模型（与构建检索库时一致）
LLM_MODEL_PATH = "/home/meixuan/models/deepseek-vl2-tiny"  # DeepSeek-VL2-Tiny 本地路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]
INPUT_SIZE = 448
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]
K = 5                                                  # 检索相似图片数量
OUTPUT_CSV = "24124053_rag.csv"                        # 输出文件
MAX_TEST = 200                                         # 最多处理测试图片数（控制时间）
USE_LOCAL_LLM = True                                   # True=本地DeepSeek-VL2, False=API

# ==================== 加载特征提取模型 ====================
print(f"加载 EVA02-L/14 特征提取模型...")
feature_model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
feature_model = feature_model.to(DEVICE).eval()

preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ==================== 加载检索库 ====================
print(f"加载检索库: {RETRIEVAL_DB_PATH}")
with open(RETRIEVAL_DB_PATH, 'rb') as f:
    db = pickle.load(f)
train_features = db['features']
train_labels = db['labels']
train_paths = db['paths']
print(f"检索库包含 {len(train_features)} 张训练图片")

# ==================== 加载 LLM（如果使用本地模型） ====================
if USE_LOCAL_LLM:
    print(f"加载 DeepSeek-VL2-Tiny 本地模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deepseek_vl2.models.processing_deepseek_vl_v2 import DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images
    import json

    # 加载 processor（使用与实验二相同的修复方式）
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
        LLM_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).cuda().eval()
    print("DeepSeek-VL2-Tiny 加载完成")

# ==================== 辅助函数 ====================
def image_to_base64(img_path):
    """将图片转换为 base64 编码"""
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def extract_feature(img_path):
    """提取单张图片的特征向量"""
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_model(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().squeeze().numpy()

def retrieve_similar(test_feat, k=K):
    """检索最相似的 K 张训练图片"""
    sims = cosine_similarity(test_feat.reshape(1, -1), train_features)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return top_indices, sims[top_indices]

def build_prompt(retrieved_labels, retrieved_sims):
    """构建发送给 LLM 的文本 Prompt"""
    prompt = "你是一位病理学专家。以下是几张H&E染色的细胞核病理图像及其类别：\n\n"
    for i, (label, sim) in enumerate(zip(retrieved_labels, retrieved_sims)):
        prompt += f"示例{i+1}：类别：{label}（相似度：{sim:.2f}）\n"
    prompt += f"\n现在请判断最后一张测试图片的类别。\n"
    prompt += f"类别选项：{', '.join(CLASS_NAMES)}\n"
    prompt += "请只回复类别名称，不要加任何解释。"
    return prompt

def local_llm_inference(prompt_text, retrieved_img_paths, test_img_path):
    """使用本地 DeepSeek-VL2-Tiny 进行推理"""
    # 所有图片路径（先放检索图片，最后放测试图片）
    all_img_paths = retrieved_img_paths.tolist() + [test_img_path]

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{prompt_text}",
            "images": all_img_paths,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(DEVICE)

    inputs_embeds = llm_model.prepare_inputs_embeds(**prepare_inputs)
    outputs = llm_model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=30,
        do_sample=False,
    )
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
    return answer

# ==================== 主流程 ====================
if not os.path.exists(TEST_DIR):
    print(f"测试集目录 {TEST_DIR} 不存在，请等待老师发布测试集后修改脚本中的 TEST_DIR 路径。")
    exit()

test_files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
print(f"发现 {len(test_files)} 张测试图片")

if MAX_TEST > 0:
    test_files = test_files[:MAX_TEST]
    print(f"处理前 {len(test_files)} 张（可通过 MAX_TEST 调整）")

predictions = []
for fname in tqdm(test_files, desc="RAG 推理中"):
    img_path = os.path.join(TEST_DIR, fname)

    # 1. 提取测试图片特征
    test_feat = extract_feature(img_path)

    # 2. 检索相似图片
    top_indices, similarities = retrieve_similar(test_feat, k=K)
    retrieved_paths = train_paths[top_indices]
    retrieved_labels = train_labels[top_indices]

    # 3. 构建 Prompt 并调用 LLM
    prompt_text = build_prompt(retrieved_labels, similarities)

    if USE_LOCAL_LLM:
        # 本地 DeepSeek-VL2 推理
        try:
            answer = local_llm_inference(prompt_text, retrieved_paths, img_path)
        except Exception as e:
            print(f"  {fname}: LLM推理失败 ({e})，使用最近邻")
            answer = retrieved_labels[0]
    else:
        # API 推理（需要自行实现）
        answer = retrieved_labels[0]  # 默认回退到最近邻

    # 4. 验证答案
    if answer in CLASS_NAMES:
        predictions.append(answer)
    else:
        # 尝试从答案中提取类别名
        matched = False
        for cls in CLASS_NAMES:
            if cls in answer:
                predictions.append(cls)
                matched = True
                break
        if not matched:
            predictions.append(retrieved_labels[0])  # 回退到最近邻

# ==================== 保存结果 ====================
df = pd.DataFrame({
    'filename': test_files,
    'label': predictions
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n提交文件已生成: {OUTPUT_CSV}")
print(f"总样本数: {len(df)}")

# 统计预测分布
for cls in CLASS_NAMES:
    count = (df['label'] == cls).sum()
    print(f"  {cls}: {count} ({100*count/len(df):.1f}%)")