#maven常用命令
---
####1. 打包命令,忽略测试
- mvn package -Dmaven.test.skip=true -Dmaven.javadoc.skip=true
####2. 编译
- mvn compile
####3. 拷贝到本地库
- mvn install