---
title: python常用命令
---

#### 1. 使用argparse传递命令行参数
- 格式

		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument(
		'-v', '--version',
		type=float, default=0.2, help='the version')
		对于bool
		parser.add_argument('--feature', dest='feature', action='store_true')
		parser.add_argument('--no-feature', dest='feature', action='store_false')
		args = parser.parse_args()
		print(args.version)
		print(args.feature)


- 调用
```
python test.py -v 0.3 --feature
```

<!--more-->
		
#### 2. 获取当前py文件路径 
	os.path.dirname(__file__)

#### 3. 判断类型
	t=[1,2]
	print(isinstance(t, list))

#### 4. 对字典排序
	#按照values从大到小排序
	for key, values in sorted(d.items(), key=lambda item:item[1]，reverse=True)
		print('key:{}/value:{}'.format(key, value))
		
#### 5. 遍历文件的两种方法
	os.listdir(path)

	for root, dirs, files in os.walk(path):
		for file in files:
			print(file)

#### 6. 排序
	a=[('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]  
	sorted(a, key=lambda k:len(k[0]), reverse=True)
	sorted(a, key=lambda k:(k[0], k[1])) # 多个属性排序

#### 7. map函数
	参数：map(func, *iterables)
	将函数作用到每个可迭代元素上
	类似的有filter函数，filter(func, *iterables)
			
#### 8. 删除文件和目录
	import os
	import shutil	
	os.remove('D:/1.txt')
	shutil.rmtree('D:/test/')

#### 9. 正则表达式
	import re
	# 替换字符串:
	re.sub('.*java', '', 'jfdlsjavafesd') # 返回fesd
	re.search(r'(\d+(\.\d*)?)\s*元', '9e12.43元').groups()[0] # 返回12.43

#### 10. 随机数
	import random
	# 0-99之间随机取10个随机数
	random.sample(range(100), 10)
	random.randint(0, 100) #生成0-100之间的一个整数
	data = range(100)
	random.shuffle(data) #打乱 return None

#### 11. 格式化字符串
	‘{:,.2f}’.format('12.9877') # 保留两位小数
	'{:>4}'.format('12') # ^、<、>分别是居中、左对齐、右对齐，后面带宽度
#### 12. 格式化时间
	import datetime
	print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
	# 在当前时间加13小时（可以负数），也可以把hours换成days、minutes
	(datetime.datetime.now()+datetime.timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S') 
