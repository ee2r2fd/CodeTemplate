import tensorflow as tf

'''
各个网络层可以定义成类或者方法，然后修改各个网络层的时候就在这个文件中进行版本的更新
'''
# def fc_layer(x):
#     # （2，8）
#     with tf.variable_scope('fc_layer', reuse=tf.AUTO_REUSE):
#         w = tf.get_variable(name='w', initializer=tf.contrib.layers.xavier_initializer(), shape=(8, 3), dtype=tf.float32)
#         b = tf.get_variable(name='b', initializer=tf.contrib.layers.xavier_initializer(), shape=(3), dtype=tf.float32)
#         x = tf.matmul(x, w)+b
#         return x

class FC_layer():
    '''
    全连接层
    '''
    def __init__(self):
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            self.w = tf.get_variable(name='w', initializer=tf.contrib.layers.xavier_initializer(), shape=(8, 3), dtype=tf.float32)
            self.b = tf.get_variable(name='b', initializer=tf.contrib.layers.xavier_initializer(), shape=(3), dtype=tf.float32)

    def fc(self, x):
        x = tf.matmul(x, self.w)+self.b
        return x