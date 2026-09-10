# 24GB 显存参数版

本目录只保留“提高超参数、复用最新训练核心”的版本。旧 `train_G24.py` 的双文本编码器 LoRA / 独立训练循环已移除。

## 文件

- `train_24.py`：CUDA 和显存检查、角色选择、应用配置、调用根目录 `train.run_training()`。
- `config_24.py`：全部24GB专属超参数与显存门槛。
- 训练核心仍是根目录 `train.py`，隔离运行由 `training_run.py` 管理。

## 使用

在项目根目录运行：

```bash
python G24/train_24.py
```

也可直接在 IDE 运行本目录入口，路径按文件位置解析。角色由根目录 `config.CHARACTER_ID` 选择；找不到对应数据目录时使用原训练入口的角色选择菜单。模型路径、角色卡、标签覆盖等继续沿用根配置。

## 默认配置

| 项目 | 值 |
|---|---|
| 分辨率 | 1024 |
| 步数 | 1500 |
| UNet LoRA rank / alpha | 64 / 64 |
| 学习率 | 1.2e-4 |
| batch / 梯度累积 | 1 / 1 |
| 精度 | fp16 |
| checkpoint 间隔 | 300 |
| 固定参考诊断间隔 | 50 |
| 比例桶预算 | 长边1088、面积不超过1024² |
| 文本编码器 | 冻结，缓存文本嵌入 |
| 梯度检查点 / 8bit Adam | 开启；优化器不可用时沿用核心的回退机制 |
| 显存门槛 | 至少22 GiB，面向标称24GB设备 |

1024/1500步/rank64只是较高容量的起点，不保证优于8GB版本；没有在本机8GB上伪称完成24GB训练验证。保持 batch=1 以适配现有可变尺寸样本，增加 batch 前需另外处理不同尺寸样本的批次拼接。

## 数据与产物保护

默认 `TRAIN_ISOLATED_RUN=True` 仅隔离预处理数据和审核日志：

- 权重：`output/24/train_models/`，与标准版分开。
- 曲线：`output/24/curves/`。
- 数据副本、代码快照、manifest 和日志：`output/24/logs/training/run_<时间>/`。

仅覆盖24GB目录内的同角色同名权重，不覆盖标准版。新权重不需要复制，设置根配置 `LORA_MODEL_SOURCE="24"`，再通过 `LORA_SELECTION_MODE="final"` 或 `"best"` 选择；设为 `"standard"` 则选择标准目录。默认保持 standard；选择24后找不到权重会报错，不会跨目录回退。explicit 模式直接使用 `LORA_WEIGHT_PATH` 的完整路径，忽略来源目录拼接。

标签可选；缺少时由 VLM 尝试生成，质量不保证，优先手动提供。隔离模式下生成的标签在数据副本中。关闭 `TRAIN_ISOLATED_RUN` 只取消源数据保护，不改变权重/曲线的统一位置。

## 与网页及推理的关系

网页训练仍调用根训练核心，并使用自己的任务参数；不会自动套用24GB配置。推理继续运行根目录 `test.py`。提高推理分辨率时，可按设备预算调整 `INFERENCE_RESOLUTION` 和 `INFERENCE_MAX_LONG`；不要仅因训练为1024就假定同时加载ControlNet后的显存也充足。
