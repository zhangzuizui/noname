# Time Series Analysis with Python
本文通过各种搬运+总结形成。如果看不了公式的话，也许你需要装一个[Github with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related),是一个chrome插件。
## Introduce
总之最近需要做KDD2017和公司的一个项目，通过各种调查研究锁定了以前都没有涉及到的领域----时间序列分析。用它的理由大概就是这届KDD和公司那个项目的数据，都不是横截面数据，换句话说就是：它们**对于时间不独立，具有时间上的相关性**。所以不能作为一个常规的回归问题来解决，这个时候想要用曲线去拟合数据的话就只能上时间序列了。
## 1. 用pandas加载和处理时间序列
在这里所用的数据是[AirPassengers data](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv). Pandas有专门用来处理时间序列对象的库能让我们快速的处理数据。

### 1.1 代码细节

```
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m') 
data = pd.read_csv('AirPassengers.csv', parse_dates='Month', index_col='Month',date_parser=dateparse)
```

这里提一下这两个参数：
1. **parse_dates**: 对于这个参数，需要传入一个list，list里面存放的是需要处理的包含date-time信息的列名；
2. **date_parser**: 这个参数其实就是需要传入一个函数，这个函数用来读取str格式的数据，返回date-time格式，这个参数默认的读取格式是 'YYYY-MM-DD HH:MM:SS'.

### 1.2 查询操作
表的查询方法和正常的操作是一样的，不过能以两种方式实现：
1. 用一个字符串来指定index: 
```
data['#Passengers']['1949-01-01']
```
2. 使用datetime库:
```
from datetime import datetime
data['#Passengers'][datetime(1949,1,1)]
```

若像查询多个字段的话，用切片操作即可，但是与常规的切片有所不同。比如a[:5]会输出index为0, 1, 2, 3, 4的内容。而data[:'1949-01-01']会输出index为'1949-01-01之前'一直到'1949-01-01'的内容(会多一个)。

使用datetime格式作为index的另一个优势是，能够按年或者按月查表，如
```
#这一句会输出1949年的所有数据
print(data['#Passengers']['1949']
```
#### P.S.
为了更加方便的操作数据，可以把DataFrame转化为Series，因为这个例子中只有**月份**和**乘客人数**两个属性，将月份作为index后就能一一对应到乘客人数。如不经转换，那么每次对DataFrame进行操作时都得多输入一个['#Passengers']比较麻烦。

### 1.3. ADF的参数细节
statsmodels.tsa.stattools.adfuller()里，有一项参数为autolag,可供传递的参数有'AIC', 'BIC'和't-stat'。

这里记录下AIC和BIC。They are sometimes used for choosing best predictor subsets in regression and often used for comparing nonnested models, which ordinary statistical tests cannot do.那么，在这里应该是使用AIC或者BIC来找子集。AIC和BIC该如何选择，看[这里](https://methodology.psu.edu/node/504).
## 2.时间序列的稳定性

如果一个时间序列的统计学属性，如平均值，方差不随时间发生变化，那么我们说这个时间序列具有稳定性。为什么要检查时间序列的稳定性？因为大多数的时间序列模型，都是建立在**时间序列是稳定的**这个假设上的。直观的说，如果时间序列随时间有一个特定的行为，那么未来极可能有相同的行为。并且，关于稳定性序列的理论，在实施时更成熟且简单。

稳定性以一个非常严苛的标准定义，但是，对于特定的目的，如果序列有随时间不变的统计学属性，我们可以假设它具有稳定性。以下为三个判定时间序列稳定的基础标准：
1. 序列的均值不能是时间的函数，换句话说序列的均值是一个常熟(该均值的含义具体参见rollingAverage)  

![图2-1](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Mean_nonstationary.png)

2. 序列的方差不为时间的函数。

![图2-2](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Var_nonstationary.png)

3. 不依赖于时间的自动协方差(不知道是啥意思？看一下[这里](https://en.wikipedia.org/wiki/Autocovariance))，即第i项和第(i+m)项的协方差不能是时间的函数(注意到下面右图的曲线，它的频率在随时间变化)。

![图2-3](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/Cov_nonstationary.png)

如果违反了以上三个标准，就需要让时间序列具有稳定性，然后尝试用随机模型来预测这个时间序列。使时间序列稳定的方法有很多，比如去趋势，差分化等。

### Random Walk
这里介绍一个时间序列中最基础的概念：Random Walk。就是说一个女孩儿站在棋盘上，我们要做的就是在女孩的初始位置的情况下预测女孩儿接下来的位置(每次只能移动到临近的8个格子中的一个)。

![Random Walk](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/RandomWalk.gif)

显而易见的是，因为每次预测都是基于上一次的结果，所以我们预测的误差会越来越大。用公式可以将这个序列表示为:
$$X(t)=X(t-1)+Er(t)$$
其中Er(t)是在时间t时的误差。这是一个完全随机的数，因为不论哪个时间我们都不知道女孩儿会往哪个方向走。

如果我们递归的将所有X聚合到一起，能够得到以下方程:
$$X(t)=X(0)+Sum(Er(0),Er(1),...,Er(t))$$

现在时间序列的方程已经得到，接下来就可以验证这个时间序列的稳定性了:

1. 其均值是否为常数？
$$E[X(t)]=E[X(0)]+Sum(E[Er(0)],E[Er(1)],...,E[Er(t)])$$
因为误差是随机的，所以它的期望为0，于是得到$E[X(t)]=E[X(0)]$，是常数
2. 方差是否为常数?
$$Var[X(t)]=Var[X(0)]+Sum(Var[Er(0)],Var[Er(1)],...,Var[Er(t)])$$
简化后得到
$$Var[X(t)]=Var[X(0)]+t*Var(Error)$$
很明显是与时间相关的，所以这不是一个稳定的时间序列(同时，如果计算这个方程的协方差，也会发现其协方差不是常数)

## 3. 时间序列的稳定性测试
让我们引入一个新的系数$\rho$，来看上面的公式会有什么变化。
$$X(t)=\rho *X(t-1)+Er(t)$$
接下来会举例的描绘出不同$\rho$值时的函数图像，以便直观的体会到$\rho$的影响。

当$\rho=0$时，

![rho=0](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho0.png)

当$\rho=0.5$时，

![rho=0.5](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho5.png)

可以看到上面的两张图，虽然周期很模糊，但本质上没有很严谨的违背稳定性的假设。但当$\rho=0.9$时，

![rho=0.9](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho9.png)

在这里依旧是能看出$X$在每经一个间隔到达极值后，扔会回到0，当然这个序列也没有显著的违背非稳定性。最后，当$\rho=1$时，

![rho=1](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/rho1.png)

这明显就是一个非稳定的情况了。为什么$\rho=1$时，在该稳定性测试中表现的如此糟糕？因为这个方程：

$$X(t)=\rho*X(t-1)+Er(t)$$

显而易见的，$\rho$越大，误差的积累就越多。换一个角度看：

$$E[X(t)]=\rho*E[X(t-1)]$$

$\rho$越大，每次操作对下一次的影响力就越大。
### Dickey-Fuller检验
上面的部分提到的就是著名的Dickey-Fuller检验，只要对上面的式子进行细微的调整就能得到Dickey-Fuller检验的形式:

$$X(t)-X(t-1)=(\rho-1)X(t-1)+Er(t)$$

这里要做的是检验一个null hypothesis:$(\rho-1)$显著性的不等于0----如果这个null hypothesis被拒绝的话，我们就得到一个稳定的时间序列。
#### P.S.
时间序列的稳定性检测，以及转化一个序列为稳定序列是时间序列建模中最关键的步骤。

## 4. 如何让一个时间序列变稳定
虽然许多时间序列模型都建立在时间序列具有稳定性这个假设下，但在实际应用中几乎没有稳定的时间序列。所以我们需要尽可能的让时间序列变得稳定。

首先让我们了解让时间序列不稳定的两个主要原因：
1. **趋势**：均值随时间变化。 如在示例代码中，乘客人数随着时间的增加呈上升趋势。
2. **季节性**：在特定时间帧的变化。如人们可能更倾向于在加薪后的那个月买车。

潜在的原则就是建模或估计序列中的这种趋势和季节性，然后从该序列中移除，以得到一个稳定的序列；接着就能让时间序列模型应用到这个序列上，得出预测值；最后将预测值转换为原始格式的数据，还原其趋势和季节性，这样就得到了我们想要的结果。

### 4.1. 估计&消除趋势
转换是一种减少趋势的好方法，比如对于乘客人数的数据明显的有一个正向的趋势，所以可以对该数据做一个能够惩罚高值数据的转换，比如取对数，开平方，开立方等。在这里对数据做对数变换：

![ts_log](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/9.-ts-log.png)

在这个示例中很容易的可以看到数据的一个增长趋势，但是对于噪声的存在表现的并不是很直观。所以我们可以用以下的技巧来估计趋势或对趋势进行建模，然后从序列中将它消除：
1. 聚合 -- 取一段时间的平均数，如月平均，周平均
2. 平滑 -- 取滑动平均
3. 多项式拟合 -- 拟合一个回归模型

接下来做一个平滑法的示例。

### 4.2. 滑动平均法消除趋势
在这个方法中，我们根据时间序列的频率取**k**个连续值的平均数。在这个示例中，我们可以取上一年的平均，如12个值。Pandas里就带有这个函数，可以直接调用：
```
moving_avg = pd.rolling_mean(ts_log,12)
plt.plot(ts_log) 
plt.plot(moving_avg, color='red')
```

![rolling mean](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/10.-smooth-1.png)

红线就是移动平均值了。需要注意的是由于我们取的是某个值的前12个值为平均值，所以最开始的11个值，它们是没有平均值的，需要把它们去除。
```
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(12)
```
![nan values](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/10.5-missing-rolling.png)

就是说的上面这些空值，将它去除掉就能得到一个新的差值序列了：

```
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
```
![diff_series](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/2.-dfuller-smooth-1.png)
能够发现此时Dickey-Fuller测试值小于临界值5%时的值----也就是说，现在的这个曲线是95%置信度的稳定序列。

但是这个方法有一个缺陷：必须严格的定义时间窗**k**，在这个示例中我们还能很简单的就能找到时间窗为一年，但是在更加复杂的情景比如股票价格，就很难得到这个**k**了。所以在这里，可以引入一个叫做'带权滑动平均'的概念：时间更近的数据权重越高。分配权重的方法有很多，常用的一个叫做'指数型带权滑动平均'，所有早期的值都会被分配一个下降因子作为权重。Pandas里面也实现了这个功能：
```
expwighted_avg = pd.ewma(ts_log, halflife=12)
plt.plot(ts_log) 
plt.plot(expwighted_avg, color='red')
```
![ewma](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/12.-smooth-2.png)

这里的参数 'halflife' 定义的是进行指数衰减的数据量。其他还能设置一些参数，如 span 或 center of mass 在[Pandas官方文档](http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions)里有写。
```
ts_log_ewma_diff = ts_log - expwighted_avg 
test_stationarity(ts_log_ewma_diff)
```

![ewma_diff](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/3.-dfuller-smooth-2.png)
可以看到使用 ewma 后，序列的置信度更可靠了(注意在这个示例中不会有缺失值)。

### 4.3. 用通法消除趋势和季节性
在4.2中讲到的滑动平均值不能应用与所有情况，尤其是具有高季节性的序列，所以这里会介绍两种其他方法来消除趋势和季节性：
1. 差分法 -- 与之前的一个特定时段的值做差
2. 分解法 -- 分别对趋势和季节性建模

#### 4.3.1 差分法
差分法就是用当前的观测值减去之前某个时间的值，作为新的序列，它很适用于消除季节性。在Python中的实现很简单：
```
ts_log_diff = ts_log - ts_log.shift() 
plt.plot(ts_log_diff)
```
![differencing](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/14.-ts-diff.png)

这似乎极大的减少了趋势，来验证一下：
```
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)
```
![1_d](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/4.-dfuller-diff.png)

好吧，进行一次差分后的效果仿佛不太好，不过没关系，我们可以紧接着对这个一次差分后的结果继续差分，这样也许会得到一个更好的结果。

#### 4.3.2. 分解法
在这个方法中，把趋势和季节性都分开建模了，使用statsmodels库很容易就能做到这一点：
```
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
```
![decomposing](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/16.-decompose.png)

趋势和季节性都分开建模了，那么检查余项的稳定性就好了：
```
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)
```
![residuals](https://www.analyticsvidhya.com/wp-content/uploads/2016/02/5.-dfuller-decompose.png)

测试结果小于1%临界值，皆大欢喜。


## reference
[A comprehensive beginner’s guide to create a Time Series Forecast (with Codes in Python)](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/)
