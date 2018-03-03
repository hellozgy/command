---
title: pandas实用命令
---

#### 1. 读csv文件
	import pandas as pd
	from pandas import DataFrame, Series
	df = pd.read_csv('./test.csv')

<!--more-->

#### 2. 删除一些列
	df.drop(['col1', 'col2'], axis=1, inplace=True)

#### 3. groupby 求平均值
	df.groupby(['PACKAGE', 'FILE']).mean().reset_index()

#### 4.对某一列进行操作
	df['col1'] = df['col1'].map(lambda x: x.replace('.', '/'))

#### 5. 选取
	df.iloc[1:4, 3:6]

#### 6. 连接
	#how默认为inner,可以为“left”，“right”，“outer”
	pd.merge(left, right, on=['key'], how='inner')
#### 7. 列重命名
	df.rename(columns={'A':'a'}, inplace = True)

#### 8.处理空
	train[COMMENT].fillna("unknown", inplace=True) #填充
	df.dropna() # 只要有空就删除某行
	df.dropna(how='all') # 全为空才删除

#### 9. 选择
	df=df[df['A']==1]
	