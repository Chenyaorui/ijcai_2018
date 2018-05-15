# ijcai_2018
ijcai_2018 复赛排名 60

比赛赛题链接：

https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.59d445580qtzFU&raceId=231647

主要思路如下：
1.用7号早上的数据作为训练和本地验证，用31-6号的数据提取历史统计特征

2.用7号当天的数据提取统计转化率等特征，用户点击行为，商品，店铺的特征，以及类目属性等的特征，但因为上午的样本分布存在差异，因此对此做了
模型权重的调整，按照线上与线下的比例调整模型的权重，或者对数据做一定处理。

3.leak特征，包括用户下一次点击的时间差，以及用户点击的时间顺序，用户是否是第一次点击，是否是最后一次点击，由此延伸出来是否用户是否第一次，
最后一次点击某商品，是否第一次点击，最后一次点击某类目等等，包括与最后一次点击的时间差。

4.lgb单模型

不足：
1.电脑只有16G，没有处理大量数据的经验，不能很好挖掘31-6号的历史行为特征

2.对用户的兴趣行为，包括类目属性和预测的类目属性，等等都没有做深入挖掘

3.lgb树模型特征选择没有太多经验，后期特征增多会导致特征冗余

4.没有系统的提取特征

5.没有做模型融合

心得体会：第一次真正的打一场比赛，从初赛的300+名，到复赛一直徘徊在一百名左右，到复赛换榜后突然开窍，成绩开始慢慢上升。从每天反向提升到开始有一丁点
感觉，这个过程有点漫长，好在一直坚持住了，也许是因为有队友和一起参赛的小伙伴一起讨论互相鼓励。可惜开窍得晚，加上最后几天各种突发情况，很多有效的方向没
时间深入挖掘，例如时间差，用户的购物兴趣等。
