# 本地运行说明

## 运行前准备

1. 准备支持本项目的 Windows + CUDA 本地 Python 环境。模型子进程由 `ENDLESS_PYTHON` 指定；未指定时使用启动 `run_local_workflow.py` 的 Python。具体依赖版本和显存/内存需求依 GPU 与 Diffusers 构建版本而异，参见根目录 README 中的当前实验环境记录。
2. 在环境变量中配置模型：`ENDLESS_QWEN_IMAGE_MODEL` 指向解压完整的 Qwen-Image-2.1；`ENDLESS_PE_I2I_MODEL` 指向完整的 Qwen-Image-2.1-PE-I2I。推理阶段设置 Hugging Face 离线变量，不会回退到在线服务。
3. 将用户素材放在本地 `input/` 与 `dataset/<角色>/`。人物卡需要 `human_complete` 配置；物品/动物卡需要 `material_isolated` 配置。卡片应记录来源和素材哈希，不能在源素材变化后继续沿用旧卡。
4. 当前分支不包含角色卡的视觉分析和服装分割 worker，也不包含用户的卡片数据；运行前须准备有效角色卡。缺卡或素材摘要不匹配时流程会停止并说明原因。

## 命令行

```powershell
python run_local_workflow.py --character <角色目录名> --kind human --input input/source.png
```

常用参数：

- `--kind human|object|animal`：按人物、物品或动物选择固定提示约束。
- `--fresh`：跳过冻结配方匹配，重新调用本地 PE-I2I 规划提示。新提示不保证与旧冻结提示等价。
- `--seeds 42 101 202`：每张输入生成并保留多个种子，不自动选优。
- `--out output/local_workflows/my_run`：显式指定这次运行的可恢复工作目录。变更原图、角色、参考、提示、模型或种子时应使用新的目录。
- `--model <path>`：覆盖 Qwen-Image-2.1 生成模型；PE-I2I 模型通过 `ENDLESS_PE_I2I_MODEL` 配置。
- `--prepare-only`：只准备计划、验证输入并落盘，不执行模型推理。
- `--technical-retries 0|1|2`：外层监督器的原生进程异常恢复上限，默认 1；只有 Windows 访问冲突触发恢复，不因画质或评分重采样。

## 运行阶段

1. 对输入、角色卡、固定参考、模型、代码和参数建立请求绑定。能匹配本地 `histories/quality_20261116_stability` 下冻结基准且完整性核验通过的人物任务，可以复用原提示嵌入；这份基准不随 GitHub 新主线发布。没有基准或无法精确匹配时使用新规划。
2. PE-I2I 本地规划器从原图和角色参考生成提示。程序核验提示图像槽位和画布绑定；规划不合规时中止任务，不静默丢弃输入或切换无图通用提示。
3. Qwen-Image-2.1 先编码图像/文字条件，再用 NF4 Transformer 进行 40 步生成。默认启用 BF16 条件编码、视觉塔 BF16、CPU 分层卸载、CPU 图像前缀 KV 缓存及 CUDA 缓存整理。
4. 保存生成图、模型阶段日志、请求和输入快照、SHA-256 摘要、恢复记录以及原图/参考/结果对比页面。

## 续跑与结果语义

- 相同 `--out` 和完全相同请求可续跑；已完整的阶段和结果会按请求与文件摘要复用。
- 变更任何输入或执行条件都不能覆写原任务；新建输出目录后重新运行。
- 外层默认仅对 `0xC0000005` 进程访问冲突恢复一次。恢复沿用同一提示、缓存、输入和种子；其他错误直接失败。
- `index.html` 是结果查看页，不是自动质量评分。`technical_completion=true` 和 `quality_status=not_evaluated` 表示技术执行结束、视觉质量仍待人工判断。
- 当前公开分支没有自动评价候选、自动选择最好种子或自动把人工看图意见变成局部编辑指令的步骤。

