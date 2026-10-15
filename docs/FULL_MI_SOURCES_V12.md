# v12：扩大MI源开发池

v11整源留一MI开发验证约51–53%，提示目前训练域迁移仍弱。现有v1池只含Cho/Dreyer且每被试每类最多40trial；本轮取消此上限，纳入旧审计清单中的Cho、Dreyer、Lee全部train/validation。原始清单test、2b、2a均不读取。

已核对原清单：train为Cho8480、Lee8800、Dreyer15992；validation分别1000、1000、2160，共37432个输入条目。计数是跨分片重复审计前数量，最终可用量以审计为准。继承Dreyer13/14排除。

Lee此前被作为外部开发评估（曾只用其5个test被试子集）；本轮明确将其原train/validation作为源开发，不再把Lee成绩称独立外部证据。以前的评估历史不抹去。SEED/SEED-VII角色不变。

统一C3/Cz/C4、200Hz、完整trial先1–30Hz滤波、取前3秒、逐通道zscore/clip15；冻结REVE时间池化为3×512。记录每条标准化信号SHA256，四卡分片完成后必须审计跨split重复/标签冲突、被试交集及完整索引；审计通过前不训练。

缓存`artifacts/mi_full_sources_v12`。后续将以整个源数据集留一进行开发选模，最终目标集不用于超参数选择。源数据扩充并不保证迁移改善，也不是新增任务。
