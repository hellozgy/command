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