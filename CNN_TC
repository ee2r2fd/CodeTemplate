'''
demo:tensorflow实现的CNN用于文本分类
'''
import tensorflow as tf
import numpy as np
import argparse
import tensorflow.contrib as tc


class CNN_TC():
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        self.train_batch_size = args.train_batch_size
        self.label_num = 35
        self.word_dim = 300
        self.epoch = args.epoch
        self.word2vec_route = args.word2vec_route
        self.filter_size = args.window
        self.lr = args.lr

        self.batch_data = tf.placeholder(shape=(None, self.max_seq_length), dtype=tf.int32)  # 占位符定义一个batch的索引数据
        self.batch_label = tf.placeholder(shape=(None, ), dtype=tf.int32)

        input = self.embed_layer()
        out = self.CNN_layer(input)
        out = self.fc_layer(out)
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_label, logits=out)  # (batch_size, )
        # 先将out经过softmax函数,然后将self.batch_label转换成onehot编码,
        # logits指的是没经过softmax或者sigmoid函数的映射到无穷区间的数据,经过了这两个函数就是映射到一定区间的概率分布
        self.batch_mean_loss = tf.reduce_mean(batch_loss, axis=0)

        # classifier_train_op = tf.train.AdamOptimizer(self.lr).minimize(batch_mean_loss)
        self.classifier_train_op = tc.opt.LazyAdamOptimizer(self.lr).minimize(self.batch_mean_loss)


    def embed_layer(self):
        '''
        将原始的一个batch的索引数据self.batch_data经过embed层转换成一个batch的特征数据self.inputs
        :return:
        inputs:一个batch的特征数据 tensor (batch_size, max_seq_length, word_dim)
        '''
        self.word_map, embed_matirx = self.load_word_embed()
        embed_matrix = tf.get_variable(name='embed_matrix', initializer=embed_matirx, trainable=False)  # 这里的embed矩阵设置为不可更新
        input = tf.nn.embedding_lookup(params=embed_matrix, ids=self.batch_data)  # (batch_size, max_seq_length, word_dim)
        # embedding_lookup不是简单的查表而是一种全连接层,输入向量是onehot,权重矩阵是embed矩阵
        # params对应的权重矩阵(embed矩阵self.embed_matrix)是可以训练更新的，训练参数个数应该是(总词数, word_dim)，但是在这里设置成了不可更新的
        return input


    def CNN_layer(self, input):
        '''
        将一个batch的特征数据inputs(经过embed层得到的)经过CNN层得到输出out
        :param input: 一个batch的特征数据 tensor (batch_size, max_seq_length, word_dim)
        :return:
        out: 卷积层的输出 tensor (batch_size, sent_dim=(max_seq_size-filter_size)/stride + 1)
        '''
        filter = tf.get_variable(name='filter', shape=(self.filter_size, self.word_dim, 1), dtype=tf.float64)
        out = tf.nn.conv1d(
            value=input,
            filters=filter,
            stride=1,
            padding='VALID'
        )
        out = tf.squeeze(out, -1)
        return out

    def fc_layer(self, input):
        '''
        将经过CNN层得到的
        :param input:卷积层的输出 tensor (batch_size, sent_dim=(max_seq_size-filter_size)/stride + 1=58)
        :return:
        out:全连接层的输出 tensor (batch_size, label_num)
        '''
        weight = tf.get_variable(name='relation_weight', shape=(58, self.label_num), dtype=tf.float64)
        bias = tf.get_variable(name='relation_bias', shape=(self.label_num, ), dtype=tf.float64)
        out = tf.matmul(input, weight)+bias  # (batch_size, label_num)
        return out


    def load_word_embed(self):
        '''
        加载embed矩阵embed_matrix和对应的词表word_map
        :return:
        embed_matrix:embed矩阵 arr (总词数, word_dim)
        word_map:词表 dict {'PAD':0,...}
        '''
        word_map = {}
        word_map['PAD'] = 0
        word_map['UNK'] = 1
        embed_matrix = []
        with open(self.word2vec_route, 'r', encoding='utf-8')as f:
            for line in f.readlines():
                line_list = line.strip().split()
                if len(line_list) != self.word_dim+1:
                    continue
                word_map[line_list[0]] = len(word_map)
                embed_matrix.append(np.array(line_list[1:], dtype=np.float32))

        mean = np.mean(embed_matrix)  # embedding矩阵中所有元素的均值
        std = np.std(embed_matrix)  # embedding矩阵中所有元素标准差
        embed_head = np.random.normal(mean, std, size=[2, self.word_dim])  # 生成pad和UNK对应的向量
        embed_matrix = np.array(embed_matrix)
        embed_matrix = np.concatenate((embed_head, embed_matrix), axis=0)  # 将pad和UNK对应的向量放在embed矩阵开头
        return word_map, embed_matrix


    def data_batcher(self):
        '''
        产生一个batch的训练数据及对应的标签
        :return:
        batch_data:一个batch的训练数据 list batch_size*(max_seq_length, )
        batch_label:一个batch的label list [batch_size, ]
        '''
        data = np.random.randint(0, 5000, [130, self.max_seq_length])  # 每个元素是一个词在词表中的索引
        # 创造130个样本
        label = np.random.randint(0, 35, [130, ])
        data_order = list(range(130))
        np.random.shuffle(data_order)

        for i in range(130//self.train_batch_size):  # 产生第i个batch的数据
            batch_data = []
            batch_label = []
            for index in data_order[i: (i+1)*self.train_batch_size]:  # 将下标为index的数据加入到当前的batch的数据中
                batch_data.append(data[index])
                batch_label.append(label[index])

            yield batch_data, batch_label


    def run_model(self, sess, saver):

        for i in range(self.epoch):
            print('Epoch: ', i+1)
            for batch_data, batch_label in self.data_batcher():
                feed_dict = {}
                feed_dict[self.batch_data] = batch_data
                feed_dict[self.batch_label] = batch_label
                loss, _ = sess.run([self.batch_mean_loss, self.classifier_train_op], feed_dict)
                print(loss)

'''
build model
2019-09-26 16:24:37.623766: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
Epoch:  1
3.636746483194225
3.518460908354998
Epoch:  2
3.630077702511205
3.5782784660159037
Epoch:  3
3.7339628158022133
3.5806558874254573
'''

import os
import random


def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    tf.set_random_seed(2019)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_seq_length",
                        default = 60,
                        type=int,
                        help="句子最大长度")
    parser.add_argument("--train_batch_size",
                        default = 64,
                        type = int,
                        help = "训练时batch大小")
    parser.add_argument("--epoch",
                        default=3,
                        type=int,
                        help="训练轮数")
    parser.add_argument("--word2vec_route",
                        default='./data/raw/word2vec.txt',
                        type=str,
                        help="预训练词向量路径")
    parser.add_argument("--window",
                        default=3,
                        type=int,
                        help="卷积核大小即卷积窗口大小")
    parser.add_argument("--lr",
                        default=0.01,
                        type=float,
                        help="学习率")
    parser.add_argument("--cuda",
                        default='0',
                        type=str,
                        help="学习率")
    args = parser.parse_args()


    tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形
    print('build model')
    gpu_options = tf.GPUOptions(visible_device_list=args.cuda, allow_growth=True)  # 选择id为FLAGS.cuda的gpu
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
    # 内存，所以会导致碎片
    # gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Graph().as_default():
        set_seed()
        sess = tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
                                  intra_op_parallelism_threads=int(multiprocessing.cpu_count() / 2),
                                  inter_op_parallelism_threads=int(multiprocessing.cpu_count() / 2)))
        # allow_soft_placement=True如果你指定的设备不存在，允许TF自动分配设备
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope('', initializer=initializer):
                model = CNN_TC(args)
            sess.run(tf.global_variables_initializer())  # Variable类型的算子，在打开会话后进行初始化。
            saver = tf.train.Saver(max_to_keep=None)
            model.run_model(sess, saver)


if __name__ == '__main__':
    main()  # 终端输入：python CNN_TC.py
