---
title: Tensorflow常用命令
---

#### 1. the difference
- **sparse_softmax_cross_entropy_with_logits:** labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1]
- **softmax_cross_entropy_with_logits:** labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
	
Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.

<!--more-->

#### 2. 配置tensorlfow
    config = tf.ConfigProto()
	# 配置gpu资源随着需求增加，不设置会占用所有gpu资源
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
		sess.run(init)


	# Creates a graph.
	with tf.device('/gpu:2'):
  		a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  		b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  		c = tf.matmul(a, b)
	# Creates a session with allow_soft_placement and log_device_placement set
	# to True.
	sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
	# Runs the op.
	print(sess.run(c))

- **allow_soft_placement:**If you would like TensorFlow to automatically choose an existing and supported device to run the operations in case the specified one doesn't exist, you can set allow_soft_placement to True in the configuration option when creating the session.
- **log_device_placement:**To find out which devices your operations and tensors are assigned to, create the session with log_device_placement configuration option set to True


#### 3. 配置超参数
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_float('learning_rate',1e-3,'Initial learning rate')
	flags.DEFINE_integer('max_step',20000,'Number of steps to run trainer')
	flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer1')
	flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer2')
	flags.DEFINE_integer('batch_size',100,'Batch Size')
	# 引用的时候直接调用：
	FLAGS.max_step
	
#### 4. cnn初始化
	weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
	biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')	
- **tf.truncated_normal:**正态分布，超过均值两个标准差的值会重新选择（截断）,stddev参数是标准差，一般设置为1/sqrt(n)，其中n为输入层的size
- b一般初始化为0

#### 5. 保存中间结果 Checkpoint & 并重新加载
	saver = tf.train.Saver()
	...
	# saver只会保存最新的5次保存结果
	saver.save(sess, FLAGS.train_dir, global_step=step)

	# reload in the future
	model = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if model and model.model_checkpoint_path:
                print(model.model_checkpoint_path)
                saver.restore(sess, model.model_checkpoint_path)
	
#### 6. 打开log日志
- tensorflow日志分5个依次严重的等级：debug,info,warn,error,fatal。tf默认是warn，
可以通过下面的语句修改log等级。设置INFO后，tf.contrib.learn训练loss将会每隔100次打印结果。
	
		tf.logging.set_verbosity(tf.logging.INFO)
		
#### 7. 日志监控 & early stopping
- 四种监视器

	Monitor|Description
	-------| -----------
	CaptureVariable | Saves a specified variable's values into a collection at every n steps of training
	PrintTensor | Logs a specified tensor's values at every n steps of training
	SummarySaver | Saves tf.Summary protocol buffers for a given tensor using a tf.summary.FileWriter at every n steps of training
	ValidationMonitor | Logs a specified set of evaluation metrics at every n steps of training, and, if desired, implements early stopping under certain conditions

- 下面这段代码每隔50步评估测试集(注意：如果要看到这些信息，要设置log等级为INFO)

        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    	test_set.data,
    	test_set.target,
    	every_n_steps=50)

	ValidationMonitor依赖于checkpoints,所以在初始化classifier时要添加RunConfig，如下所示，表示每隔1秒保存依次checkpoint

		classifier = tf.contrib.learn.DNNClassifier(
	    feature_columns=feature_columns,
	    hidden_units=[10, 20, 10],
	    n_classes=3,
	    model_dir="/tmp/iris_model",
	    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

- 自定义

		validation_metrics = {
	    "accuracy":
	        tf.contrib.learn.MetricSpec(
	            metric_fn=tf.contrib.metrics.streaming_accuracy,
	            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
	    "precision":
	        tf.contrib.learn.MetricSpec(
	            metric_fn=tf.contrib.metrics.streaming_precision,
	            prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
	    "recall":
	        tf.contrib.learn.MetricSpec(
	            metric_fn=tf.contrib.metrics.streaming_recall,
	            prediction_key=tf.contrib.learn.PredictionKey.CLASSES)}
		
		# 最后三行设置early_stopping，分别代表评估指标，最小化还是最大化，early_stopping的步数
		validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
	    test_set.data,
	    test_set.target,
	    every_n_steps=50,
	    metrics=validation_metrics,
	    early_stopping_metric="loss",
	    early_stopping_metric_minimize=True,
	    early_stopping_rounds=200)

- 可视化

		tensorboard --logdir=./tmp/iris_model/ --host=127.0.0.1 --port=6006

#### 8. 可视化tensor

	def variable_summaries(var):
	  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	  with tf.name_scope('summaries'):
	    mean = tf.reduce_mean(var)
	    tf.summary.scalar('mean', mean)
	    with tf.name_scope('stddev'):
	      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
	    tf.summary.scalar('stddev', stddev)
	    tf.summary.scalar('max', tf.reduce_max(var))
	    tf.summary.scalar('min', tf.reduce_min(var))
	    tf.summary.histogram('histogram', var)
	# 调用
	with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
	...
	# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
	                                      sess.graph)
	...
	summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, step)
可视化命令参照第7条。

#### 9. 打印所有可训练变量
	with tf.Session() as sess:
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        print(k, v)
		
#### 10. 将稀疏矩阵转换为稠密矩阵
	tf.sparse_tensor_to_dense(labels, default_value=-1)
 注：稀疏矩阵，比如indice=[[0,1],[1,2]],values=[10,32],
意思就是，2*3矩阵[[0,10,0],[0,0,32]]，默认将稀疏部分补0.
#### 11. tensorflow实现双向双层RNN
		双层双向RNN
        fw_cell = bn_rnn.BNGRUCell(self.config.n_hidden, self.training_placeholder)
        bw_cell = bn_rnn.BNGRUCell(self.config.n_hidden, self.training_placeholder)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=fw_cell, output_keep_prob=self.drop_out_placeholder)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(cell=bw_cell, output_keep_prob=self.drop_out_placeholder)
        fw_cells = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * 2)
        bw_cells = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * 2 )

        _, state = tf.nn.bidirectional_dynamic_rnn(fw_cells,bw_cells,inputs,sequence_length=self.seq_length_placeholder,dtype=tf.float32)
        outputs = tf.concat((state[0][0], state[0][1], state[1][0], state[1][1]), 1)
        outputs = tf.reshape(outputs, [self.config.batch_size, 4 * self.config.n_hidden])
	



