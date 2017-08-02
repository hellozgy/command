#Tensorflow常用命令#

---
####1. the difference####
- **sparse_softmax_cross_entropy_with_logits:** labels must have the shape [batch_size] and the dtype int32 or int64. Each label is an int in range [0, num_classes-1]
- **softmax_cross_entropy_with_logits:** labels must have the shape [batch_size, num_classes] and dtype float32 or float64.
	
Labels used in softmax_cross_entropy_with_logits are the one hot version of labels used in sparse_softmax_cross_entropy_with_logits.
####2. 配置tensorlfow
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
####3. 配置超参数
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_float('learning_rate',1e-3,'Initial learning rate')
	flags.DEFINE_integer('max_step',20000,'Number of steps to run trainer')
	flags.DEFINE_integer('hidden1',128,'Number of units in hidden layer1')
	flags.DEFINE_integer('hidden2',32,'Number of units in hidden layer2')
	flags.DEFINE_integer('batch_size',100,'Batch Size')
	# 引用的时候直接调用：
	FLAGS.max_step
