---
title: mysql常用命令
---
#### 1.导出到文件：
	select count(1) from table  into outfile '/tmp/test.xls';
#### 2.导入外部数据
	LOAD DATA LOCAL INFILE "D:\\train.csv" INTO TABLE test.`power_data` FIELDS TERMINATED BY ',';
#### 3.删除一列
	alter TABLE 表名 drop 列名
####4.删除表记录，保留表结构
	TRUNCATE TABLE tt
####5.更新某字段为查询值,并将null值换为0
	update a inner join b on a.bid=b.id set a.x=b.x,a.y=ifnull(b.y,0) ;
####6.插入数据为查询结果
	INSERT INTO TableA(c1,c2,c3)   
	SELECT TableB.c1,TableB.c2,TableB.c3  
	FROM TableB  
####7.添加一个字段
	alter TABLE tt add price decimal(m,n) default null;
####8.方差
	select variance(price) from tt group by id;
	标准差
	STD:n、STDDEV_SAMP:n-1
	正则表达式，使用rlike / not rlike / REGEXP   / not REGEXP 
	SELECT * FROM test.`month_data` WHERE MONTH RLIKE  '[0-9]'
####9.更新
	update test set a=2, b=3;
