import tensorflow as tf

'''
各个网络层可以定义成类或者方法，然后修改各个网络层的时候就在这个文件中进行版本的更新
'''

class Loss_layer():
    '''
    损失层：用于训练
    '''
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_loss(self, x, label):
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=label))
        _classifier_train_op = tf.train.AdadeltaOptimizer(rho=0.95, epsilon=1e-6).minimize(loss)
        return loss, _classifier_train_op