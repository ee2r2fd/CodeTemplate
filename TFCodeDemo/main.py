import os
import random
import numpy as np
import tensorflow as tf
from model.model import Model
import multiprocessing


def load_data(is_training):
    if is_training:
        sent_reps = np.random.randn(4, 8)  # 训练数据共4条
        label = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 标签共三种
        return sent_reps, label
    else:
        sent_reps = np.random.randn(4, 8)  # 训练数据共4条
        label = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 标签共三种
        return sent_reps, label

def data_batcher(is_training):
    '''
    产生一个batch的训练或者测试数据的生成器
    :return:
    '''
    if is_training:
        sent_reps, label = load_data(is_training)
        sample_num = len(label)

        for i in range(int(sample_num/FLAGS.batch_size)):
            batch_sent_reps = sent_reps[i * FLAGS.batch_size: (i+1) * FLAGS.batch_size]
            batch_label = label[i * FLAGS.batch_size: (i+1) * FLAGS.batch_size]
            yield batch_sent_reps, batch_label

    else:
        sent_reps, label = load_data(is_training)
        sample_num = len(label)

        for i in range(int(sample_num / FLAGS.batch_size)):
            batch_sent_reps = sent_reps[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            batch_label = label[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
            yield batch_sent_reps, batch_label

def train(sess, saver, model):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)

    for i in range(FLAGS.epoch):
        print('epoch ' + str(i) + ' starts...')
        for j, (batch_sent_reps, batch_label) in enumerate(data_batcher(FLAGS.is_training)):
            feed_dict = {
                model.sent_reps: batch_sent_reps,
                model.label: batch_label
            }

            result = sess.run([model.loss, model.train_op], feed_dict=feed_dict)

            print('epoch:' + str(i) + '  batch:' + str(j) + '  loss:' + str(result[0]))
        if (i + 1) % 2 == 0:  # 每隔一个epoch保存一次模型
            print('saving model...')
            path = saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my_model'), global_step=i)
            # 使得保存的模型的名称前缀为my_model-i  其中i为当前epoch的编号
            print('have saved model to ' + path)

def test(sess, saver, model):
    for i in range(FLAGS.epoch):  # 用训练时候第i个epoch保存的模型来测试
        if not os.path.exists(os.path.join(FLAGS.checkpoint_dir, 'my_model' + '-' + str(i) + '.index')):
            continue
        print('start testing checkpoint, epoch =', i)
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, 'my_model' + '-' + str(i)))  # 恢复第epoch个模型保存的静态图和相应的变量
        pred_label = []
        for j, (batch_sent_reps, batch_label) in enumerate(data_batcher(FLAGS.is_training)):
            feed_dict = {
                model.sent_reps: batch_sent_reps,
                model.label: batch_label
            }

            result = sess.run([model.pred_label], feed_dict=feed_dict)
            batch_pred_label = result[0]  # (batch_size, num_classes)
            pred_label.append(batch_pred_label)  # batch_num*(batch_size, num_classes)
        pred_label = tf.concat(pred_label, axis=0)  # (总样本数，num_classes)
        sent_reps, label = load_data(False)
        total = len(label)
        num = 0
        for k, l in enumerate(label):
            if tf.convert_to_tensor(l) == pred_label[k]:
                num += 1
        acc = num/total

        print('epoch:' + str(i) + '  acc:' + str(acc))

def set_seed():
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(2019)
    random.seed(2019)
    tf.set_random_seed(2019)

tf.app.flags.DEFINE_string('cuda', '0', 'gpu id')
tf.app.flags.DEFINE_integer('is_training', 1, 'training or not')
# tf.app.flags.DEFINE_boolean('is_training', False, 'training or not')  # 注意flags的bool类型是有bug的，不能正常使用
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch size')
tf.app.flags.DEFINE_integer('epoch', 10, 'batch size')
tf.app.flags.DEFINE_integer('num_classes', 3, 'number of class')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'path to store checkpoint')
FLAGS = tf.app.flags.FLAGS

def main(_):
    gpu_options = tf.GPUOptions(visible_device_list=FLAGS.cuda, allow_growth=True)  # 选择id为FLAGS.cuda的gpu
    sess = tf.Session(
        config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
                              intra_op_parallelism_threads=int(multiprocessing.cpu_count() / 2),
                              inter_op_parallelism_threads=int(multiprocessing.cpu_count() / 2)))
    with sess.as_default():
        set_seed() # 设置随机种子
        model = Model(FLAGS) # 初始化模型
        model.create_gragh()  # 建立静态图
        sess.run(tf.global_variables_initializer())  # 初始化静态图中的变量
        saver = tf.train.Saver(max_to_keep=None)  # 用于保存训练好的模型
        if FLAGS.is_training:
            train(sess, saver, model)
        else:
            test(sess, saver, model)

if __name__ == '__main__':
    tf.app.run()  # 解析命令行参数，并把参数传到main(_)中，并运行main
    # 如果要将解析的参数传到其他方法并调用，比如调用test(),则此句改为tf.app.run(test())
    # 训练模型,终端输入:python main.py --is_training 1 然后就执行tf.app.run()也就解析了命令行参数并运行了main
    # 测试模型,终端输入:python main.py --is_training 0
'''
训练：
epoch 0 starts...
epoch:0  batch:0  loss:1.926471
epoch:0  batch:1  loss:1.9544325
epoch 1 starts...
epoch:1  batch:0  loss:0.9700546
epoch:1  batch:1  loss:5.081016
saving model...
have saved model to ./checkpoint/my_model-1
epoch 2 starts...
epoch:2  batch:0  loss:6.500753
epoch:2  batch:1  loss:4.9534597
epoch 3 starts...
epoch:3  batch:0  loss:0.4952203
epoch:3  batch:1  loss:3.6167808
saving model...
have saved model to ./checkpoint/my_model-3
epoch 4 starts...
epoch:4  batch:0  loss:5.768976
epoch:4  batch:1  loss:3.2138755
epoch 5 starts...
epoch:5  batch:0  loss:2.132122
epoch:5  batch:1  loss:5.8616905
saving model...
have saved model to ./checkpoint/my_model-5
epoch 6 starts...
epoch:6  batch:0  loss:0.48785323
epoch:6  batch:1  loss:2.0845277
epoch 7 starts...
epoch:7  batch:0  loss:1.6211299
epoch:7  batch:1  loss:2.272787
saving model...
have saved model to ./checkpoint/my_model-7
epoch 8 starts...
epoch:8  batch:0  loss:3.785808
epoch:8  batch:1  loss:2.150423
epoch 9 starts...
epoch:9  batch:0  loss:2.501152
epoch:9  batch:1  loss:2.7589002
saving model...
have saved model to ./checkpoint/my_model-9

测试：
start testing checkpoint, epoch = 1
epoch:1  acc:0.0
start testing checkpoint, epoch = 3
epoch:3  acc:0.0
start testing checkpoint, epoch = 5
epoch:5  acc:0.0
start testing checkpoint, epoch = 7
epoch:7  acc:0.0
start testing checkpoint, epoch = 9
epoch:9  acc:0.0
'''