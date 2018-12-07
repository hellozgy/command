---
title: git常用命令
---

#### 1. gitignore
- 忽略java目录下除src之外的其他目录

		java/*
		java/src

<!--more-->

#### 2. 配置初始化
	git config --global user.name 'zgy'
	git config --global user.email '1243268625@qq.com'

#### 3. 生成ssh key:
	ssh-keygen -t rsa -C "1243268625@qq.com"

#### 4. 添加删除远程库
	git remote add origin git@github.com:hellozgy/repo_name.git
	git remote remove origin

#### 5.撤销更改
- 还没提交到暂存区,又对本地库修改，将本地库文件回退到跟版本库相同的状态

		git checkout -- file.txt
- 已经提交到暂存区，又对本地库修改，将本地库文件回退到跟暂存区相同的状态
 
		git checkout -- file.txt
- 撤销暂存区的修改， 分两步：

		# 将暂存区的修改放回本地库
		git reset HEAD file.txt
		# 将本地库的修改回退到跟版本库相同的状态
		git checkout -- file.txt
- 已经提交到版本库， 想回退到某个版本
		
		# 查看提交commit id
		git log 
		git reset --hard commit_id
#### 6. git stash：清除更改
#### 7. 删除远程文件
	git rm -r --cached .idea/
#### 8. 清除本地更改，强制更新
	git fetch --all
	git reset --hard origin/master

#### 9. 修改远程库地址
	git remote set-url origin git@gitlab.com:hellozgy/nmt2.git
#### 10.设置git显示中文
	git config --global core.quotepath false



