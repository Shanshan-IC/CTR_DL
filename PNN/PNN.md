## 应用背景

实验也是基于展示广告的CTR预估场景，本文通过product layer层探索交叉特征。

## PNN

![PNN](pic/pnn.png)

PNN 模型的最终预测值为$$hat{y} = \delta(W_3l_2 + b_3)$$，$$W_3$$和$$b_3$$是输出层的参数，$$l_2$$是隐藏层的输出结果，$$ \delta(x)$$是sigmoid激活函数，其中两个隐藏层

$$l_2 = relu(W_2l_1 + b_2)$$
$$l_1 = relu(l_z + l_p + b_1)$$
(第一个隐藏层是全连接product layer，包含线性linear signals $$l_z$$ 和quadratic signals $$l_p$$)

神经元的inner product为： $$A\bigodot B = \sum_{\substack{i, j}} A_{i, j} B_{i,j}$$，即
1. 先对A,B做element_wise乘积——res
2. 对res求和得到scaler
3. $$l_p$$和$$l_z$$进行z和p变换：
	$$l_z = (l_z^1, l_z^2, ..., l_z^n,..., l_z^{D_1}), l_z^n = W_z^n \bigodot z$$
	$$l_p = (l_p^1, l_p^2, ..., l_p^n,..., l_p^{D_1}), l_p^n = W_p^n \bigodot p$$
其中，$$W_z^n$$和$$W_p^n$$是product layer的权重。

## 参考文献

- Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.

