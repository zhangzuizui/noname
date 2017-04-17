# 关于移动平均值
详情参见[这里](https://en.wikipedia.org/wiki/Moving_average)

在统计学中，移动平均值(moving average, rolling average or running average)是分析数据点的一种计算方法。

这个方法是通过生成一连串的整个数据集的不同子集的平均数来进行的。

在pandas里面，有个东西叫rolling_mean，这个方法？类？需要给它两个？参数，一个自然是full data set，另外一个就是窗口大小。

移动平均就常常与时间序列数据一起使用，来铺平短期波动，highlight长期趋势或者周期。这个短期和长期的阈依赖具体应用情景而定，相应的，移动平均的参数也会被设置。

在数学上，移动平均是一种卷积，然后一般有三种平均方式。
SMA，CMA和WMA，看上面wiki吧。