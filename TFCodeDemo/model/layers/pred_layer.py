import tensorflow as tf

'''
各个网络层可以定义成类或者方法，然后修改各个网络层的时候就在这个文件中进行版本的更新
'''

class Pred_layer():
    '''
    预测层：用于测试
    '''
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pred(self, x):
        prob = tf.nn.softmax(x)  # (batch_size, num_class)
        pred_label_index = tf.argmax(input=prob, axis=1)  # (batch_size, )
        pred_label = tf.one_hot(indices=pred_label_index, depth=self.num_classes, dtype=tf.float32)  # (batch_size, num_classes)

        return pred_label