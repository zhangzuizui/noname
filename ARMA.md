# 两个时间序列模型
## 1. ARMA
首先要明白的是ARMA(Auto-Regression and Moving average)模型只能应用在稳定的时间序列上。然后接下来会分开描述 AR 和 MA 这两个模型。
### 1.1. AR
AR模型的公式如下:
$$X(t)=\alpha X(t-1)+error(t)$$
这是一个标准的**AR(1)**模型，这个1指的是X(t)只与其上一个数据，即X(t-1）有关。其大致图像如下：![AR](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/AR1.png)

### 1.2. MA
MA的公式如下:
$$X(t)=\beta error(t-1)+error(t)$$
其图像为：![MA](https://www.analyticsvidhya.com/wp-content/uploads/2015/02/MA1.png)
### 1.3. 两个模型的不同
显而易见的，AR的图像呈指数型下降，而MA实际上就是几个分段函数，类似于折线图，其 X(t) 和 X(t-n) 的相关性，总是为0。而在AR中，X(t) 和 X(t-n) 的相关性，随着n的减小而增大。
### 1.4. 模型的使用

