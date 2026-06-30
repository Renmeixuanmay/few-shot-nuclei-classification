import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# ==================== 配置 ====================
TRAIN_DIR = "/home/meixuan/data/train_few_shot"
INPUT_SIZE = 448
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD  = [0.26862954, 0.26130258, 0.27577711]
OUTPUT_PATH = "MixUp_augmentation_example.png"
CLASS_NAMES = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4"]

# ==================== 预处理与 MixUp ====================
base_preprocess = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def mixup_data(x, alpha=0.4):
    """对两个张量进行 MixUp 混合"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    mixed_x = lam * x[0:1] + (1 - lam) * x[1:2]
    return mixed_x, lam

def tensor_to_pil(tensor):
    """将归一化后的 Tensor 转回 PIL Image"""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, MEAN, STD):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1)
    tensor = tensor * 255
    tensor = tensor.permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(tensor)

# ==================== 生成对比图 ====================
fig, axes = plt.subplots(len(CLASS_NAMES), 3, figsize=(10, 14))
axes[0, 0].set_title("Original Image", fontsize=12)
axes[0, 1].set_title("MixUp Result", fontsize=12)
axes[0, 2].set_title("Paired Image", fontsize=12)

for i, class_name in enumerate(CLASS_NAMES):
    class_dir = os.path.join(TRAIN_DIR, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # 随机选两张不同的同类别图片
    img1_name, img2_name = random.sample(images, 2)
    img1 = Image.open(os.path.join(class_dir, img1_name)).convert('RGB')
    img2 = Image.open(os.path.join(class_dir, img2_name)).convert('RGB')

    # 转为 tensor 并 MixUp
    t1 = base_preprocess(img1).unsqueeze(0)
    t2 = base_preprocess(img2).unsqueeze(0)
    mixed, lam = mixup_data(torch.cat([t1, t2], dim=0), alpha=0.2)

    # 转回 PIL 显示（统一放大到 256×256）
    img1_show = img1.resize((256, 256), Image.BICUBIC)
    img2_show = img2.resize((256, 256), Image.BICUBIC)
    mixed_show = tensor_to_pil(mixed.squeeze(0)).resize((256, 256), Image.BICUBIC)

    # 绘制
    axes[i, 0].imshow(img1_show)
    axes[i, 0].axis('off')
    axes[i, 0].set_ylabel(class_name, fontsize=10)

    axes[i, 1].imshow(mixed_show)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(img2_show)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"MixUp 增强示例图已保存至: {OUTPUT_PATH}")