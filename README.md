# few-shot-nuclei-classification
Few-shot nuclei classification on 32x32 H&amp;E patches (250 samples, 5 classes)
# 少样本细胞核病理图像分类

## 项目简介
本仓库为《机器学习》课程期末大作业的代码实现，目标是在每类仅 50 张训练样本的条件下构建 5 分类细胞核病理图像分类器。最终方案 ConvNeXt-L + MixUp + Linear Probe 取得了 Macro F1 均值 0.9745 的交叉验证成绩。

## 环境依赖
Python 3.10+, PyTorch 2.x, timm, torchvision, scikit-learn, pandas, numpy, matplotlib, seaborn

## 目录结构
- linear_probe_mixup/：七种预训练模型的 MixUp + Linear Probe 实验
- augmentation_ablation/：增强策略消融实验
- peft_experiments/：LoRA 和 Adapter 微调实验
- clip_baseline/：CLIP Zero-shot / KNN / LLM+RAG 实验
- other_experiments/：扩散模型合成数据、t-SNE 可视化等

## 复现说明
1. 将训练数据放入 `data/train_few_shot/` 目录
2. 安装依赖：`pip install -r requirements.txt`
3. 运行各目录下的脚本即可复现实验

## 模型来源
所有预训练模型均来自 timm / HuggingFace / torchvision，详见报告附录 A。

## 许可证
本仓库代码仅用于学术研究目的。
