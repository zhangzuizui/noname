# Dickey-Fuller test
Dickey-Fuller测试是测试了一个null hypothesis----unit root是否在自回归模型中。alternative hypothesis因使用的测试的版本的不同而不同，但通常是平稳性或趋势-平稳性测试。

[*根据这里*](http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html)可知，对于ADF，$H_0$=有unit root，$H_1$=没有unit root

测试方法如下:

对于一个AR(1)模型:
$$y_t=\rho y_{t-1}+u_t$$
其中，$y_t$是目标变量，$t$是时间，$\rho$是系数，$u_t$是误差项，如果$\rho=1$则单位根存在(也有说$|\rho|\geq1$)。
AR(1)模型中的这个1，表示的是当前的数据只与上一个数据相关（t和t-1）.

这个回归模型可以写为
$$\Delta y_t=(\rho-1)y_{t-1}+u_t$$

这样在模型中对unit root的估计和测试就变为了检测$\delta=0$.因为测试是使用的残差项而不是原始数据，所以不能使用标准t-分布来提供临界值，于是这个统计量$t$有一个特定的分布，叫做Dickey-Fuller分布。

检测的三个主要版本:
1. $\Delta y_t=\delta y_{t-1}+u_t$
2. $\Delta y_t=\delta y_{t-1}+u_t+a_0$(带偏置项)
3. $\Delta y_t=\delta y_{t-1}+u_t+a_0+a_1t$(带偏置项和确定的时间趋势)

每个版本根据其样本大小，都有自己的临界值。在每种情况中，null hypothesis都是:存在unit root,$\delta =0$.因为测试的统计敏感性很低，所以常常不能分辨真unit-root过程($\delta=0$)和近似unit-root过程($\delta$接近0)，这称作'near observation equivalence'(近观察等值)问题。

另外，还有一个扩展的Dickey-Fuller测试，叫做augmented Dickey-Fuller test(ADF)，它移除了时间序列中的全部结构性影响(自相关)然后用同样的过程测试。

## Null hypothesis
[wiki传送门](https://en.wikipedia.org/wiki/Null_hypothesis)
假设衡量的两个现象之间没有关系，或者组间无关联(就是两个或者两个以上的东西，他们之间没关联呗)。如果拒绝了这个假设的话，就意味着blabla之间有关联。一般写作$H_0$

## Alternative hypothesis
艾总就是null hypothesis的反面了，写作$H_1$.

## Statistical power
或者说Statistical sensitivity, 是$H_1$为真时，拒绝$H_0$的概率，或者说$H1$为真时，接受$H1$的概率。

$power=P(rejectH_0|H_1\ is\ true)$

随着这个概率的递增，发生Type **||** error的机会就会减少，或者叫false negative rate,同样的还有一个Type **|** error,也就是false positive rate.
