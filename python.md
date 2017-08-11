# python常用命令
----
####1. 使用argparse传递命令行参数
- 格式

		import argparse
		parser = argparse.ArgumentParser()
		parser.add_argument(
		'-v', '--version',
		type=float, default=0.2, help='the version')
		# 对于bool
		parser.add_argument('--feature', dest='feature', action='store_true')
     	parser.add_argument('--no-feature', dest='feature', action='store_false')
		args = parser.parse_args()
		print(args.version)
		print(args.feature)
- 调用

		python test.py -v 0.3 --feature
####2. 获取当前路径 
	os.path.abspath('.')
	#或者
	os.system('pwd')

####3. 判断类型
	t=[1,2]
	print(isinstance(t, list))
