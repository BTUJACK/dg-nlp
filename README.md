## “达观杯”文本智能处理挑战赛的尝试

### 评分标准

评分算法
binary-classification
评分标准 
采用各个品类F1指标的算术平均值，它是Precision 和 Recall 的调和平均数。

### Day1 逻辑回归
1. 用 CountVectorizer 类将文本中的词语转换为词频矩阵
2. 利用逻辑回归进行训练
3. 分数0.732344

4. 尝试tf-idf 分数 0.76871


**TODO:**
1. Split training set 
2. 在本地大概算分数，挑好的提交
3. 
