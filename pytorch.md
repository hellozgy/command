---
title: pytorch常用命令
---

#### 1. 形状改变
	x=torch.randn(2,3)
	x.view(-1, 6)
	
<!--more-->

#### 2. gather(dim, index)
将维度dim的index位置的元素抽取出来聚合在一起。
下面代码将x按照维度2抽出最大的两个数的位置，并按照从小到大排序

	x=Variable(torch.randn(2,3,4))
	index = x.topk(2, dim = 2)[1].sort(dim = 2)[0] 
	x.gather(dim, index)

##### 3. detach()
将Variable从图出抽取出来，也就是不计算梯度，可以用于词向量固定不变的情况。

#### 4. permute(*args)
按照维度变换形状，比如下面代码将第一维和第二维调换位置。可以用于batch和seq维度的转换。

	v=Variable(torch.randn(2,3,4))
	v.permute(1,0,2)
#### 5. nn.Sequential()
可以定义一个线性变换的序列，比如
	
	self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.hidden_size*2*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )



