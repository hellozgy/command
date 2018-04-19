---
title: numpy常用命令
---

####1. 生成随机数
	# 0-1
	np.random.rand(2, 3)
	# 高斯分布
	np.radnom.randn(2,3)
	# 生成一定范围内的整数
	np.random.randint(0, 10, size=(3, 4))
	# 生成m-n的序列
	np.random.randint(0, 10, size=(4,))
	np.arange(0, 12).reshape(3,4)
	

#### 2. 类型转换
	a=np.random.rand(2,3)# float64
	a.astype(np.float32)
