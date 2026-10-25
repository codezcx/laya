# 检索方法、范围与证据分级

检索日期：2026-10-25（UTC）。目标为本年度已录用工作启发的、小算力且能贡献问题定义的研究方向。明确排除训练基础模型、VLM 方法堆叠作为主路线；允许把相关工作作为新颖性冲突证据。

## 目录来源与真实覆盖量

| 来源 | 获取方式 | URL 去重入口数 | 首轮关键词匹配 |
|---|---|---:|---:|
| NeurIPS 2026 Downloads | 官方 HTML，保留 /virtual/2026/poster 或 oral 链接 | 9146 | 325 |
| ECCV 2026 AcceptedPapers | 官方 HTML，相同链接规则 | 2834 | 81 |
| ICML 2026 Downloads | 官方 HTML，相同链接规则 | 6628 | 221 |
| ICLR 2026 Papers | 官方 HTML，相同链接规则 | 5468 | 126 |
| AISTATS，PMLR v300 | 官方论文集 div.paper 与 abs 链接 | 588 | 36 |
| COLT，PMLR v336 | 同上 | 197 | 5 |
| UAI，PMLR v337 | 同上 | 330 | 34 |
| 合计 | 不是跨 track 去重后的官方录用数 | 25191 | 828 |

首轮题目匹配正则（忽略大小写）：

```text
conformal|missing|abstention|selective|identifiab|sensor|calibrat|change.point|stopping|label.shift|test.time.adapt|partial.monitor|active.feature|acquisition|annotation|delayed
```

后续补充词包括 granularity、boundary、timestamp、misalignment、synchronization、montage、observability、partial labels、coarsening、ranking、reference、interval censoring。对基础模型相关命中做临时标题过滤以便阅读，但这些自动过滤结果没有当作人工排除判定，也没有用来估计某领域的热度。

**54 篇定向官方页面阅读**来自主题筛选、后续关键词与邻近工作追踪；不是声称 828 篇都完成了摘要审查，也不是随机抽样。完整标题及来源见 papers.csv / papers.json。不同会议 track 没有逐篇核验，因此官方 poster 条目只能支持会议收录，不能额外支持“主会主轨/Oral/最佳论文”等称谓。CVPR 另有官方搜索索引核验的 1 篇主会、1 篇 workshop，未计入上述 54。

## 检索记录

sources/search_01.json 到 search_10.json 保留查询参数、日期和工具返回文本。sources/read_*.json 保留已知链接的查看与访问失败信息。最初的宽泛年份/官网导航搜索仅用于定位目录，未用于定量统计。

主要检索族：

- 2026 官方会议 + missing/conformal/partial identification/calibration。
- temporal annotation uncertainty + model comparison/ranking/boundary。
- EEG + montage/reference/identifiability。
- symmetry/quotient + conformal prediction。
- time series + clock drift/temporal misalignment/certified robustness。
- ECG multi-expert delineation、QT Database two annotators、TAS released features。

数据库/主来源包括会议官网、PMLR、CVF、OpenReview、arXiv，以及 PhysioNet/Zenodo 数据原始发布页。搜索服务用于发现来源；第三方整理页仅当导航，不作为确认录用的最终依据。旧论文用于查重，不混入 2026 接收统计。预印本明确单列，不用 arXiv 日期推断会议录用。

## 原文检查与访问限制

下载结果在 sources/fulltext_fetch.json。共尝试 14 份 PDF，成功 9 份，5 份 OpenReview 返回 403/验证页。没有绕过访问检查；通过官方摘要继续工作。

8 份定向原文检查：ECCV coordinate singularities、ECCV source-free model ranking、AISTATS mask-conditional CP、AISTATS partial-label learnability、UAI ternary tests、COLT coordinate corruption regression、Kuboyama model-choice 预印本、GRAFT-EEG 预印本。重点看问题设置、主张范围、相关假设及局限；不是完整证明重审。

ECCV multi-hypothesis TTA PDF 成功提取但只使用摘要层面结论。其余论文以官方摘要为主，明确不据此认定某定理适用于任意数据机制。

已检查出的实质限制包括：mask-conditional CP 的似然比/支撑条件；几何覆盖结果的局部近似与分组覆盖范围；无标签模型排序的任务对齐限制；partial-label reconstructibility 与未知生成机制下实际学习的区别；GRAFT-EEG 的限制潜在模型与算子扰动假设。

CVPR 官方网站的直接请求出现 TLS/403，工具搜索缓存仍返回 CVF 原始摘要和 bibliographic record；未将该缓存描述成实时全文获取。初始 COLT 卷号探索曾访问 v291，发现年份/会议不符后改为官方目录确认的 **v336**；当前 catalog 使用 v336，不引用错误卷号。

## 数据核验与未完成事项

QTDB 以官方 .q2c / .qt2 文件说明为准：第二位专家的标注只有 11 个记录。NeurIPS 新 ECG 数据的“由两位专家核验”不能推断原始双标注可用。Zenodo API 元数据显示同名数据包为开放访问、CC-BY-NC-4.0，大小 307783032 字节，创建者仍为 Anonymous；这还不足以确认其与论文的对应关系。先通过 HTTP Range 读取 ZIP 尾部 262144 字节，再读取尾部 1048576 字节，获得与 ZIP 结束记录一致的完整 6995 条目录。另读取开头 56110 字节，解压 README 和 map_file.csv，只保存说明及字段汇总，不保存逐人元数据。核实有 2732 个 ECG 记录、1527 个 `.ii` 文件；doctor1–doctor7 是异常诊断标签，说明中没有发布独立专家多套波形边界。未解析 `.ii` 文件或原始波形，因此该检查仍不能穷尽文件内部字段。对应来源记录为 zenodo_zip_directory.json 和 zenodo_zip_metadata.json。

## 证据与推断分开

| 等级 | 能支持什么 | 不能支持什么 |
|---|---|---|
| 官方列表/论文集 | 标题、年份、会议收录 | 复现成功、主张正确、所有 track 标签 |
| 官方摘要 | 作者研究的问题及声称结果 | 省略假设后的普适保证 |
| 定向全文 | 实际设置、定理条件、局限 | 本次未逐行审核的全部证明正确性 |
| 本次解析证明 | 明确数学模型内的结论 | 新颖性、现实适用性、投稿价值 |
| 小规模数值交叉核验 | 实现公式与穷举的一致性 | 一般数学证明或真实数据效果 |

## 可复查材料与工作树边界

- sources/*_catalog.json：七个目录的标题/URL 提取。
- sources/papers/*.json：54 篇官方页面文本和论文/代码链接。
- sources/manifest.json：本地源文件大小与 SHA256；哈希只保证快照可辨认，不保证内容真伪。
- sources/screening_counts.json：自动标题匹配计数及正则。
- papers.csv / papers.json：人工主题解释、证据深度与选题含义。
- PROBLEM_DEFINITIONS.md：当前三个候选的完整基础证明、限制和未证明的主结果。
- check_math.py / math_checks.json：独立穷举核对 DP 与线性区间端点。只使用合成小例子，不是训练代码。

原始 HTML/PDF/全文文本仅作本地下载证据，在本目录 .gitignore 中排除；未删除任何既有数据、模型或实验文件。没有改动仓库已有训练入口、配置、README 或运行协议，已有用户修改保持原样。本调研是一套当前研究文档，不是新的并行实验实现。

## 工具流程披露

使用 literature-review、scientific-brainstorming 与 PDF 技能。Parallel CLI 与对应 API key 不可用，因此使用可用网页搜索与直接 HTTP 读取。没有额外安装或索取凭据。没有召集独立评审者；反向质疑为同一助手完成，不能称为独立复核。

scientific-schematics 的外部图像服务凭据不可用；为了使图形可复核，使用标准 Matplotlib 绘制由四帧反例精确计算的科学图，而不是让生成图像承担数值论证。它不是 PRISMA 图，不虚构人工筛选数目。

运行了技能附带 DOI 核验器，但它仅检查文中 DOI，且 DOI 解析/网络错误不等于论文不存在；不能拿“DOI 检查通过”冒充 54 篇的接收核验。主要核验来自逐条官方 URL 与本地目录对应。过程参考 [Scientific Agent Skills (2026)](https://doi.org/10.48550/arXiv.2609.00065)。
