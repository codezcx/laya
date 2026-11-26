# LEAF Tower → MedJev：13库共享监督训练

当前主线将原始EEG信号替换MedJev的图像输入，不运行个人适配或元学习。

```text
EEG → 发布版LEAF信号Tower → 256→2560→2560连接器
    → Qwen3.5前23个完整语言层
    → 来自第32层的Q/K候选匹配＋零初始化残差MLP
    → 候选类别概率
```

文本包含任务说明、问题、候选选项及MedJev标记；没有MPNet、FiLM、Q-Former、图像encoder或逐token生成。原始Qwen3.5权重与MedJev医学适配权重不同；本轮仅加载前者。

## 训练和数据

沿用此前13库的 `datasets/leaf13/manifest.json`，不重新划分或重复预处理。train用于梯度；原test用于开发期监控和选checkpoint；原valid暂不forward，且历史上已被观察，不能称为新盲测。

2040步、有效batch16、四卡同步数据并行，每卡4条。前340步只训练新连接器；之后训练连接器、Tower全参、语言LoRA r16/alpha32、标记embedding及决策读出。学习率和采样预算继承上一轮，不替换成MedJev医学训练配方。

## 运行

```bash
laya/.venv-eeg/bin/python -m pytest tests -q
laya/.venv-eeg/bin/python -m jev_train.run --start
```

控制器依次等待权重完整、检查分片、验证数据和真实模型梯度、执行四卡train-only smoke，再从全新初始化启动正式训练。下载最多等待24小时；任何检查失败都不会自动启动正式训练。训练期间不要修改受hash保护的代码或配置。

权重目录由 `configs/jev13_train.json:backbone_path` 指定，当前使用用户的 `../MedJev/Qwen3.5-4B`。控制器不下载、不覆盖权重；代码/权重/日志不混合提交。

- [固定协议](docs/PROTOCOL.md)
- [当前状态与结果](docs/RESULTS.md)
- [模型接入](jev_train/medjev.py)
- [配置](configs/jev13_train.json)
- [上游源码与MIT许可](vendor/MedJev/UPSTREAM.md)
- [历史归档](docs/ARCHIVE.md)

旧个人LoRA与MAML源码见Git `c46748c`、归档提交 `dffd139`，只在独立worktree检查或复跑。全部ignored数据、checkpoint、缓存、环境与日志保留。
