# relationExtraction

本项目的中文实体关系抽取，是基于人力资源领域的关系抽取，其主要工作包括：[基于模板匹配的无监督关系抽取](<https://github.com/Erisu0014/relationExtraction/tree/master/pattern-relation>)、[基于CNN实现的有监督关系抽取](https://github.com/Erisu0014/relationExtraction/tree/master/CNNmodel)。

## 模板匹配的无监督关系抽取

由于领域词和实体有少许错误，及定义的模板相对简单，无监督关系抽取效果达到的可预期如下图所示。

![](https://puu.sh/DFiy8/84a89e7ef7.png)

【注】：由于对于抽取结果的划分是外行人做的，所以相关精准率可能稍有偏差，实际效果并没有这么低。



## 基于CNN实现的有监督关系抽取

基于TextCNN实现的有监督关系抽取

![](https://puu.sh/DFiDQ/fc7e14232c.png)

由于数据量太小，所以效果并不理想，但方法上应没有太大过错。

本项目还基于[NELL理论](https://puu.sh/DFiGD/97190e6f91.pdf)进行了[模型自优化](https://github.com/Erisu0014/relationExtraction/blob/master/CNNmodel/train_more.py)和关系预测，是以哪种输入集预测以及是进行自优化还是预测需要对于主函数进行相应的注释（这很容易做到）。

自优化简易算法流程图

![](https://puu.sh/DFiKG/1693e04d09.png)

## 结语

本科生毕业设计，甲方数据源给的太少，人力资源相关领域爬虫获取到的数据源有限，没有足够的人力做支持，难以得到良好的实验结果，同时，TextCNN结构设计的过于简单，也是导致最终结果差的一个原因。而以上多是因为自己对NLP的认知不足。











