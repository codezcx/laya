# 本地 AAAI 项目的 EEG RL checkpoint

目标项目位于 `/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss`，包含基于 Qwen3-VL 的 EEG 后训练实验。它与 `mne_data/cere/CerebraGloss-3B` 中的公开 Qwen2.5-VL-3B 权重是不同对象。

2026-10-14 已检查配置、索引与权重分片是否存在；随后已验证 8B step_60 的八卡决策训练与恢复：

| 底座 | HF checkpoint 根目录（相对下述 output） | 存在的 steps | hidden size | decoder 层数 | 每个 checkpoint 分片 / 文件大小 |
|---|---|---|---:|---:|---|
| Qwen3-VL-8B | `eeg_rl_hf_8b_v3render/` | 20, 40, 60, 80, 100, 120, 140 | 4096 | 36 | 4 / 17.53 GB |
| Qwen3-VL-32B | `eeg_rl_hf_32b/` | 40, 60, 80 | 5120 | 64 | 14 / 66.71 GB |

共同 output 路径：`/inspire/hdd/global_user/zhangchenxi-253108310322/CerebraGloss/DataEngine/output`。上述 HF 索引引用的分片均存在。这不等于重新验证了模型效果，也没有按最终测试集选择最优 checkpoint。

训练来源可核对：

- `DataEngine/train_32b_pipeline.sh`：Stage1 对齐 SFT → Stage2 指令 SFT（v3 图像）→ GRPO；base 为 `models/Qwen3-VL-32B-Instruct`。
- `DataEngine/run_eeg_grpo_8b.sh`：从 SFT checkpoint 启动，`algorithm.adv_estimator=grpo`，使用 verl。
- `DataEngine/paper/train_32b_8gpu_HANDOFF.md`：记录 verl actor checkpoint 合并为上述 HF 导出的路径。
- `DataEngine/eval_step60_big.sh`、`DataEngine/_eval_32b_fast.sh`：引用 8B/32B 的具体 HF checkpoint。

## 输入图像需匹配已有 RL 模型

本地训练使用 `DataEngine/render_images_v3.py`：19 通道，200 Hz，10 秒，固定跨通道幅值尺度，负电位向上；12×8 英寸、110 DPI PNG、线宽 0.6；**绘图函数本身不额外滤波**。代码中的 `SCALE_UV=100` 与行距 2.6、倍率 1.3 共同对应相邻通道 200 µV 间距；没有公开 Bench 脚本中的红色刻度条。

当前 `tusz_sft/render.py` 和 `pilot_filtered/` 复现的是公开 CerebraGloss-Bench 绘图，并额外使用 0.5–70 Hz 滤波。它们与本地 AAAI 的 v3 输入不同。使用已有 8B/32B RL checkpoint 继续训练时，应先对齐本地 v3 图像及其 NPY 生成流程，不能直接声称当前图片已匹配该 checkpoint。

## 与 JEV 的关系

可以把上述 RL checkpoint 作为底座，增加 JEV 风格的决策输出，再训练新的 LoRA：8B 二分类头为 4096→2，32B 为 5120→2。任务需要时可扩展为 24 槽位，但分类监督本身不能训练未提供标签的其他槽位。

现有 `laya/JEV-27B-VL` 是 Qwen3.5 架构。尽管它与 Qwen3-VL-32B 都有 5120 hidden size 和 64 层，两者 attention、MLP、词表和视觉融合结构不同，因此不能直接互换完整主干权重或原 JEV LoRA。需要迁移的是决策训练方式，并在选定的 RL 底座上重新初始化/训练相应适配参数。

已有 `jev_smoke_filtered/` 只验证了原 JEV-27B 的两步训练；之前依据该运行给出的单卡/八卡时间估算，不是本地 8B/32B RL 模型的实测速度。

## 当前实现

`train.py` 已切换到 `train_decision.py`；默认使用上述 8B step_60。`render_v3.py` 已对齐本地预处理和绘图。八卡训练、恢复、保存后预测一致性检查已通过。实验安排见 [README.md](README.md)。原 JEV-27B 入口保留在 `train_legacy.py`。
