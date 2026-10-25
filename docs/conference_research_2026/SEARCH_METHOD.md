# 当前检索方法与证据边界

更新：2026-10-25。当前目标是核实 EEG 可信性问题的现实性、明确定义并查重；旧边界方向已撤回。

## 本轮证据

1. 读取当前仓库 README、PROTOCOL、RESULTS、任务目录、2a 裁窗与 LEAF 归一化实现，以及相邻 EEG-FM-Benchmark 的数据构建器。仅只读检查，没有导入/运行训练代码。路径、行号、SHA256 保存在 sources/eeg_trust/local_code_evidence.json。
2. 读取 BCI IV-2a 原始说明与 PhysioNet 运动任务说明。区分 event-based 裁窗、标签来源和实际神经反应。
3. 抓取 MOABB 官方结果表，保存结构化表格，并从相同左右手 MI 任务的行列计算描述性排名反转。明确 ROC-AUC、within-session 评估、标准差含义及非 checkpoint 跨库迁移。未用汇总标准差声称显著性。
4. 从已提取的 2026 官方会议目录重新筛选 EEG、neural decoding、task specification、interpretability 等词，定向读取 7 个官方页面并保存正文/链接。它们是摘要层证据，不是全部定理重审。
5. 读取 Latent Alignment、OSPDIM、ThinkNet、EEG conformal、Visual cues 的原始全文 HTML，定向检查上下文定义、评价协议与相关结果。未逐行审核全部证明。Latent Alignment 的 arXiv 版与已发表版分开处理；ThinkNet 的摘要/HTML/第三方索引有标题或数值差异，不把版本差异混成同一精确结果。
6. 2025 preprocessing multiverse 的指定 arXiv HTML 返回 404；使用出版社原文搜索结果与作者机构记录核验相关段落，不把失败请求当全文成功。
7. 一般 TTA、conformal、数据投毒、SzCORE 等检索用于新颖性排除。搜索命中不等于采纳论文结论；第三方页面只作导航，报告优先指向原始论文与官方结果。

sources/eeg_trust/search_log.json 保留 14 组搜索工具输出与两组读取输出；最初的宽泛搜索不用于定量覆盖声明。sources/eeg_trust/ 另存网页正文、MOABB 表格、计算结果和本地代码证据。

## 不支持的结论

- 没有做当前本地受控实验来证明“上下文预算导致排名反转”。
- 不能把本地 bundled preprocessing 的改善归因于某一项归一化操作。
- 没有证据证明所有 EEG 模型不可信，也不把所有模型说成端到端。
- 已完成普通受损率认证到 LTT 的直接归约，排除该版本的新颖性；这不是对全部安全适应理论的穷尽审查。
- 作者在 arXiv 声称接收，与独立核对官方接收目录是不同证据等级。
- 真实伪迹、数据漂移和标签分歧的存在，不自动构成新的研究问题。
- 有完整数学推导，不自动意味着非平凡主定理或现实适用。

## 先前会议目录仍可用于查重

七个目录共提取 25,191 个 URL 入口：NeurIPS 9,146，ECCV 2,834，ICML 6,628，ICLR 5,468，AISTATS 588，COLT 197，UAI 330。它们不是已核定的主会独立论文总数，也不是逐篇阅读量。

首轮 828 个关键词命中不等于全部人工审查；先前定向阅读 54 篇官方页面，列表留在 PAPERS.md / papers.csv / papers.json 作为背景索引。本轮新增来源在当前 REPORT 与 sources/eeg_trust 中独立标明，不擅自修改先前计数。CVPR 的先前覆盖受限，未声称完成全目录阅读。

旧推荐、边界数学核验和图形可在 Git fae0972 查看。按照仓库规则，工作树中不继续保留已放弃方向的实现/图形；没有删除忽略的下载全文、数据、日志、checkpoint、环境或第三方安装。现有训练代码与用户修改未改变。

## 流程

使用 scientific-critical-thinking 技能和此前 literature-review 的来源记录方法。初轮无独立评审者；后续按 scientific-brainstorming 的独立生成与非提出者复核步骤使用三个 AI 评审，记录于 independent_review.json。它们不是外部专家或独立实证复现。流程参考保留在 REPORT；它不是科研结论证据。

## 后续查重与否定结果

- 定向读取 LTT 的定义和一般过程，按用户级样本、受损损失与预算参数逐项归约，撤下普通“受损率认证”候选。
- 读取 Matched-Input Estimates Differ in Sign Across Architectures 全文的实验设计、结果与限制，排除把匹配输入比较当作首次提出的问题。未独立复现论文数值。
- 核验 EEG-Arena 原始摘要；全文 HTML 因长度限制失败。不得把摘要核验升级为全文审阅。
- 本轮搜索记录在 sources/eeg_trust/novelty_followup_search.json；其中第三方页面与 gauge/reference 检索命中仅用于导航，不据此认定方法有效或 novelty 穷尽。逐项判断见 sources/eeg_trust/novelty_audit.json。
- 归约是本次分析；未启动新训练，也没有把初步文献审查说成已取得新的科研贡献。

## 本地机制初检

2026-10-25 新增 scripts/diagnose_input_factors.py，完成空间参考×归一化范围的四组冻结编码器对照，只使用七名原训练被试。事前约束、逐用户结果及不能支持的结论见 INPUT_FACTOR_DIAGNOSIS.md。四组共 28 个固定配方探针拟合；对匹配/固定分类头的 56 组预测数组独立重算 BA、翻转率和概率合法性。基线特征与已存原管线特征逐元素一致；输入基线与现有加载器一致；参数/buffers、源文件、输入 IDs 与产物哈希均核验。未进行模型/正则选择，不将源集合探索当独立确认或 novelty。资源检查采用 get-available-resources，流程引用已保留在 REPORT。

## 纵向候选的针对性查重

先独立生成六项候选，再查重并交叉复核。参考等价性、通道缺失拒判、通用 cue/伪迹审计等普通版本被排除。对“个体化选择”进行了第二次更细的审查：区分已有统计工具与尚未回答的实证估计量。主代理核对了 2606.24394 §2.5、2607.22778 的方法与 p.29 限制，以及跨会话选择论文 §2.8；2607 HTML 返回 404 后成功使用原始 PDF，而非把失败请求当全文阅读。另由评审追溯 2024 个体分类器选择与较早可靠性研究。

候选结论为 bounded-search plausible empirical gap，不是首创证明；已知近邻、未覆盖之处和仍需追溯的范围写在 repeatability_prior_matrix.json / PROBLEM_DEFINITIONS.md。检索记录见 repeatability_search_log.json。NEMAR Yang2025 的元数据请求成功；SHU 页面与原 Figshare 密码要求已区分，未联系数据作者。只检查本地 B03T/E 会话结构，未运行新的跨日分类/偏好试验。
