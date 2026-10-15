# v12：扩大MI源开发池

v11整源留一MI开发验证约51–53%，提示目前训练域迁移仍弱。现有v1池只含Cho/Dreyer且每被试每类最多40trial；本轮取消此上限，纳入旧审计清单中的Cho、Dreyer、Lee全部train/validation。原始清单test、2b、2a均不读取。

已核对原清单：train为Cho8480、Lee8800、Dreyer15992；validation分别1000、1000、2160，共37432个输入条目。计数是跨分片重复审计前数量，最终可用量以审计为准。继承Dreyer13/14排除。

Lee此前被作为外部开发评估（曾只用其5个test被试子集）；本轮明确将其原train/validation作为源开发，不再把Lee成绩称独立外部证据。以前的评估历史不抹去。SEED/SEED-VII角色不变。

统一C3/Cz/C4、200Hz、完整trial先1–30Hz滤波、取前3秒、逐通道zscore/clip15；冻结REVE时间池化为3×512。记录每条标准化信号SHA256，四卡分片完成后必须审计跨split重复/标签冲突、被试交集及完整索引；审计通过前不训练。

缓存`artifacts/mi_full_sources_v12`。后续将以整个源数据集留一进行开发选模，最终目标集不用于超参数选择。源数据扩充并不保证迁移改善，也不是新增任务。

## 审计后预定训练

四组，均seed42、3000step、batch96、lr1e-4、相同空间锚点与SEED划分：

1. capped_two：Cho/Dreyer原train中每被试每类最多40trial，Lee原train+validation作整域开发验证，普通头。
2. full_two：Cho/Dreyer完整train，同样Lee开发验证，普通头。与组1比较训练数据量。
3. full_three：Cho/Dreyer/Lee完整train，三者原validation作同域新被试验证，普通头。
4. full_three_text：与组3相同输入/划分，文本sigmoid。

组1/2跨域验证与组3/4同域验证口径不同，不能直接比较选模分数排名。组1/2另记录同域选模checkpoint。固定源比较后再决定下一步；本轮控制器不运行目标评估。

控制器`python scripts/run_full_sources.py`：等待已知提取PID完成→跨分片审计→通过才启动四组。日志与状态在`artifacts/full_sources_v12_controller`。失败不自动重启或覆盖已有结果。

四组完成后的探索性评估：完整报告四组固定checkpoint在2b与SEED-VII的成绩，不据目标挑组。复用同预处理的v6/v1冻结目标REVE缓存。Lee只作为开发域，其验证结果不是最终独立结果。目标此前已暴露，不能称新盲测；SEED-VII仍为三类子集，2b仍有encoder预训练暴露。
