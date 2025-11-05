### 项目概览
- 本仓库包含 `cifar_regressor/`，用于在 CIFAR-100 上进行分类（coarse 20 类；以及层级版 coarse+fine 100 类）。
- 已实现 ResNet/ViT 编码器；ResNet 分支可选 CBAM 注意力；层级版支持 FiLM 调制。

### 目录结构（关键）
- `cifar_regressor/model/`
  - `cifar_coarse_regressor.py`：单头模型（coarse 20 类）；接口：`CifarCoarseRegressor`
  - `hierarchical_regressor.py`：层级模型（coarse 20 + fine 100）；接口：`CifarHierarchicalRegressor`
  - `cbam.py`：CBAM 注意力模块（仅 ResNet 分支）
- `cifar_regressor/train/`
  - `train_coarse.py`：coarse-only 训练脚本
  - `train_hierarchical.py`：层级训练脚本（coarse+fine，多任务损失）
- `cifar_regressor/tools/`
  - `evaluate_coarse.py`：coarse-only 模型的测试评测（top1/top5/每类/混淆矩阵）
  - `evaluate_hierarchical.py`：层级模型的测试评测（coarse+fine 指标）
- `cifar_regressor/demo/`
  - `run_demo.py`：从测试集中随机取样，打印预测并保存结果图片
- `cifar_regressor/config/`
  - `coarse_default.json`：coarse-only 示例配置
  - `hierarchical_default.json`：层级版示例配置

### 模型接口
- `from cifar_regressor import CifarCoarseRegressor, CifarHierarchicalRegressor`

- CifarCoarseRegressor（单头，20 类）
  - 主要参数：`encoder_name`（`resnet18/resnet34/resnet50/vit_small_patch16_224`）、`pretrained_backbone`、`use_cbam`（ResNet 可用）、`hidden_features`、`dropout_p`
  - forward：`logits, probs = model(images)`
  - 采样：`model.sample_topk(x=images, k=5, num_samples=1)` 或 `from cifar_regressor.utils import top_k_sample`

- CifarHierarchicalRegressor（层级：coarse 20 + fine 100）
  - 主要参数：`encoder_name`、`pretrained_backbone`、`use_cbam`（仅 ResNet）
    `use_film`、`film_hidden`、`film_use_probs`、`hidden_features`、`dropout_p`
  - forward：返回字典 `{"coarse_logits", "coarse_probs", "fine_logits", "fine_probs"}`

说明：
- ResNet 与 ViT 编码器均支持；ViT 通过 timm（`vit_small_patch16_224`）加载预训练；CBAM 只在 ResNet 分支启用。
- 所有训练与评测脚本会从 checkpoint 的 `config` 中还原模型结构，保证推理一致。

### 数据
- CIFAR-100（Python 版）目录：`/data/litengmo/ml-test/cifar-100-python`（含 `train/test/meta`）

### 训练
- coarse-only（绝对路径示例，指定 7 号卡）
```bash
conda activate hsmr
python3 /data/litengmo/ml-test/cifar_regressor/train/train_coarse.py \
  --config /data/litengmo/ml-test/cifar_regressor/config/coarse_default.json \
  --gpu 7 \
  --print-config
```
- 层级版（coarse+fine）
```bash
conda activate hsmr
python3 /data/litengmo/ml-test/cifar_regressor/train/train_hierarchical.py \
  --config /data/litengmo/ml-test/cifar_regressor/config/hierarchical_default.json \
  --gpu 7 \
  --print-config
```
- 日志/权重
  - 每次训练会在 `checkpoint_dir/<YYYYmmdd_HHMMSS>/` 下生成 `config.json`、`train_log.json`、TensorBoard `tb/`，并同步 `last.pth/best.pth`

### 评测
- coarse-only 模型
```bash
conda activate hsmr
python3 /data/litengmo/ml-test/cifar_regressor/tools/evaluate_coarse.py \
  --dataset_root /data/litengmo/ml-test/cifar-100-python \
  --checkpoint_dir /data/litengmo/ml-test/cifar_regressor/checkpoints/coarse_resnet50 \
  --output_root /data/litengmo/ml-test/cifar_regressor/test \
  --batch_size 256 --num_workers 4 \
  --gpu 7
```
- 层级模型（coarse+fine 指标一起输出）
```bash
conda activate hsmr
python3 /data/litengmo/ml-test/cifar_regressor/tools/evaluate_hierarchical.py \
  --dataset_root /data/litengmo/ml-test/cifar-100-python \
  --checkpoint_dir /data/litengmo/ml-test/cifar_regressor/checkpoints/hier_vit_small \
  --output_root /data/litengmo/ml-test/cifar_regressor/test \
  --batch_size 256 --num_workers 4 \
  --gpu 7
```

### Demo（随机样本推理与可视化）
```bash
conda activate hsmr
python3 /data/litengmo/ml-test/cifar_regressor/demo/run_demo.py \
  --dataset_root /data/litengmo/ml-test/cifar-100-python \
  --output_dir /data/litengmo/ml-test/cifar_regressor/demo/outputs \
  --checkpoint /data/litengmo/ml-test/cifar_regressor/checkpoints/coarse_resnet18_cbam/best.pth
```

### 备注
- `encoder_name` 支持：`resnet18/resnet34/resnet50/vit_small_patch16_224`
- `pretrained_backbone: true` 将下载并缓存预训练权重（首次较慢，缓存路径 `/data/litengmo/.cache/torch/`）