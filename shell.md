---
title: Shell 常用命令
---

#### 1. 查看端口并杀死相关进程
	netstat   -ano|findstr  8080 
	taskkill  /pid  6856  /f 
#### 2. 查找字符串并显示行号
	grep -n '查找我' 文件名
#### 3. 清理显存
	fuser -v /dev/nvidia*
#### 4. scp
	scp -P 服务器端口 本机文件 服务器用户名@服务器ip:服务器文件夹




