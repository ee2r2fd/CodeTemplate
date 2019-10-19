import tensorflow as tf
from model.layers.fc_layer import FC_layer
from model.layers.loss_layer import Loss_layer
from model.layers.pred_layer import Pred_layer

class Model():
    def __init__(self, FLAGS):
        # 超参
        self.flags = FLAGS

        # 待传入数据
        self.sent_reps = tf.placeholder(dtype=tf.float32, shape=(None, 8), name='sent_reps')
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, self.flags.num_classes), name='label')

        # 初始化网络层
        self.fc_layer = FC_layer()
        self.loss_layer = Loss_layer(self.flags.batch_size)
        self.pred_layer = Pred_layer(self.flags.num_classes)

    def create_gragh(self):
        '''
        定义静态图
        :return:
        '''
        self.logits = self.fc_layer.fc(self.sent_reps)  # (2, 2)(batch_size, 2)
        if self.flags.is_training:
            self.loss, self.train_op = self.loss_layer.get_loss(self.logits, self.label)
        else:
            self.pred_label = self.pred_layer.pred(self.logits)  # (batch_size, num_classes)