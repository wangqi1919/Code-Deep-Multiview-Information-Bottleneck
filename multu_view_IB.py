import numpy as np
#import matplotlib.pyplot as plt
import math
import tensorflow as tf
import scipy.io as sio
import sys
from sklearn.utils import shuffle
import os

def train(Xtrain1, Xtrain2, ytrain, Xtest1, Xtest2, ytest, iterr, result_dir):
    num_examples, d1 = Xtrain1.shape
    _, d2 = Xtrain2.shape
    tf.reset_default_graph()
    class_num = 10
    #Turn on xla optimization
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.InteractiveSession(config=config)

    x1 = tf.placeholder(tf.float32, [None, d1], name='x1')
    x2 = tf.placeholder(tf.float32, [None, d2], name='x2')
    y = tf.placeholder(tf.int64, [None], name='y')
    one_hot_labels = tf.one_hot(y, class_num)

    layers = tf.contrib.layers
    ds = tf.contrib.distributions

    def encoder1(x, nhidden1, nhidden2, nhidden3, noutput):
        net = layers.relu(x, nhidden1)
        net = layers.relu(net, nhidden2)
        net = layers.relu(net, nhidden3)
        params = layers.linear(net, noutput)
        output_half = noutput/2
        mu, rho = params[:, :output_half], params[:, output_half:]
        encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
        return encoding

    def encoder2(x, nhidden2, noutput):
        net = layers.relu(x, nhidden2)
        params = layers.linear(net, noutput)
        output_half = noutput/2
        mu, rho = params[:, :output_half], params[:, output_half:]
        encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
        return encoding

    def fusion(view1, view2, nfoutput):
        net = layers.relu(tf.concat([view1, view2], 1), nfoutput)
        paramsf = layers.linear(net, nfoutput)
        outputf_half = nfoutput/2
        muf, rhof = paramsf[:, :outputf_half], paramsf[:, outputf_half:]
        fusion_view = ds.NormalWithSoftplusScale(muf, rhof - 5.0)
        return fusion_view

    def decoder(fusion_sample):
        net = layers.linear(fusion_sample, class_num)
        return net

    prior = ds.Normal(0.0, 1.0)
    nhidden1 = 50
    nhidden2 = 50
    nhidden3 = 50
    noutput = 40
    nfoutput = 40
    average_times = 10

    with tf.variable_scope('encoder'):
        encoding1 = encoder1(x1, nhidden1, nhidden2, nhidden3, noutput)
        encoding2 = encoder2(x2, nhidden1, noutput)

    with tf.variable_scope('fusion'):
        z1 = encoding1.sample()
        z2 = encoding2.sample()
        z = fusion(z1, z2, nfoutput)

    with tf.variable_scope('fusion', reuse=True):
        z1 = encoding1.sample(average_times)
        z1 = tf.reduce_mean(z1, 0)
        z2 = encoding2.sample(average_times)
        z2 = tf.reduce_mean(z2, 0)
        z = fusion(z1, z2, nfoutput)

    with tf.variable_scope('decoder'):
        logits = decoder(z.sample())
    with tf.variable_scope('decoder', reuse=True):
       many_logits = decoder(z.sample(average_times))

    class_loss = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels) / math.log(2)

    ALPHA = 9e-5
    BETA = 1e-3

    info_loss1 = tf.reduce_sum(tf.reduce_mean(
        ds.kl(encoding1, prior), 0)) / math.log(2)

    info_loss2 = tf.reduce_sum(tf.reduce_mean(
        ds.kl(encoding2, prior), 0)) / math.log(2)

    total_loss = class_loss + ALPHA * info_loss1 + BETA * info_loss2

    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, 1), y), tf.float32))
    avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), y), tf.float32))
    IZY_bound = math.log(10, 2) - class_loss
    IZY_bound = tf.identity(IZY_bound, name = 'IZY_bound')
    IZX_bound1 = tf.identity(info_loss1, name = 'IZX_bound1')
    IZX_bound2 = tf.identity(info_loss2, name = 'IZX_bound2')
    accuracy = tf.identity(accuracy, name = 'accuracy')
    avg_accuracy = tf.identity(avg_accuracy, name = 'avg_accuracy')

    batch_size = 64

    batch_num = int(num_examples / batch_size)
    global_step = tf.contrib.framework.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                               decay_steps=2 * batch_num,
                                               decay_rate=0.97, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate, 0.5)

    ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
    ma_update = ma.apply(tf.model_variables())

    saver = tf.train.Saver()
    saver_polyak = tf.train.Saver(ma.variables_to_restore())

    train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                       global_step,
                                                       update_ops=[ma_update])

    tf.global_variables_initializer().run()

    def evaluate(type):
        if type == 'train':
            IZY, IZX1, IZX2, acc, avg_acc = sess.run([IZY_bound, IZX_bound1, IZX_bound2, accuracy, avg_accuracy],
                                                     feed_dict={x1: Xtrain1, x2: Xtrain2, y: ytrain})
        else:
            IZY, IZX1, IZX2, acc, avg_acc = sess.run([IZY_bound, IZX_bound1, IZX_bound2, accuracy, avg_accuracy],
                                          feed_dict={x1: Xtest1, x2: Xtest2, y: ytest})

        return IZY, IZX1,IZX2, acc, avg_acc, 1 - acc, 1 - avg_acc


    for epoch in range(50):
        idx = np.random.permutation(num_examples)
        Xtrain1_shuffle = Xtrain1[idx, :]
        Xtrain2_shuffle = Xtrain2[idx, :]
        ytrain_shuffle = ytrain[idx]
        for step in range(batch_num):
            sess.run(train_tensor, feed_dict={x1: Xtrain1_shuffle[step * batch_size: (step + 1)*batch_size, :], x2: Xtrain2_shuffle[step * batch_size: (step + 1)*batch_size, :], y: ytrain_shuffle[step * batch_size: (step + 1)*batch_size]})
        print "Training: {}: IZY={:.2f}\tIZX1={:.2f}\tIZX2={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
            epoch, *evaluate('train'))
        print "Testing: {}: IZY={:.2f}\tIZX1={:.2f}\tIZX2={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(
            epoch, *evaluate('test'))
        sys.stdout.flush()

    if not os.path.exists(result_dir + '/iter' + str(iterr)):
        os.makedirs(result_dir + '/iter' + str(iterr))

    savepth = saver.save(sess, result_dir + '/iter'+str(iterr) + '/model')
    sess.close()
def load_data(validation, iterr):
    # load data
    data = sio.loadmat('data/synthetic_data/iter' + str(iterr) + '.mat')
    Xtrain1 = data['X1_train']
    Xtrain2 = data['X2_train']
    ytrain = data['ytrain']
    if validation:
        Xtest1 = data['X1_val']
        Xtest2 = data['X2_val']
        ytest = data['yval']
    else:
        Xtest1 = data['X1_test']
        Xtest2 = data['X2_test']
        ytest = data['ytest']
    return Xtrain1, Xtrain2, np.squeeze(ytrain), Xtest1, Xtest2, np.squeeze(ytest)

if __name__ == "__main__":
    train_acc = []
    test_acc = []
    train_avg_acc = []
    test_avg_acc = []
    train_IZY = []
    test_IZY = []
    train_IZX1 = []
    test_IZX1 = []
    train_IZX2 = []
    test_IZX2 = []
    train_error = []
    test_error = []
    train_avg_error = []
    test_avg_error = []
    result_dir = 'synthetic/multiview'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = open(result_dir + '/result.txt', 'w')
    for iterr in range(1, 6):
        # validation
        Xtrain1, Xtrain2, ytrain, Xtest1, Xtest2, ytest = load_data(True, iterr)
        train(Xtrain1, Xtrain2, ytrain, Xtest1, Xtest2, ytest, iterr, result_dir)
        sess = tf.Session()
        saver = tf.train.import_meta_graph(result_dir + '/iter'+str(iterr) + '/model.meta')
        saver.restore(sess,tf.train.latest_checkpoint(result_dir + '/iter' + str(iterr) + '/'))
        # testing
        Xtrain1, Xtrain2, ytrain, Xtest1, Xtest2, ytest = load_data(False, iterr)
        graph = tf.get_default_graph()
        IZY_bound = graph.get_tensor_by_name('IZY_bound:0')
        IZX_bound1 = graph.get_tensor_by_name('IZX_bound1:0')
        IZX_bound2 = graph.get_tensor_by_name('IZX_bound2:0')
        accuracy = graph.get_tensor_by_name('accuracy:0')
        avg_accuracy = graph.get_tensor_by_name('avg_accuracy:0')
        x1 = graph.get_tensor_by_name('x1:0')
        x2 = graph.get_tensor_by_name('x2:0')
        y = graph.get_tensor_by_name('y:0')
        IZY, IZX1, IZX2, acc, avg_acc = sess.run([IZY_bound, IZX_bound1, IZX_bound2, accuracy, avg_accuracy],
                                                     feed_dict={x1: Xtrain1, x2: Xtrain2, y: ytrain})
        print "Training: IZY={:.2f}\tIZX1={:.2f}\tIZX2={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(IZY, IZX1, IZX2, acc, avg_acc, 1-acc, 1-avg_acc)
        train_acc.append(acc)
        train_avg_acc.append(avg_acc)
        train_IZY.append(IZY)
        train_IZX1.append(IZX1)
        train_IZX2.append(IZX2)
        train_error.append(1-acc)
        train_avg_error.append(1-avg_acc)
        IZY, IZX1, IZX2, acc, avg_acc = sess.run([IZY_bound, IZX_bound1, IZX_bound2, accuracy, avg_accuracy],
                                          feed_dict={x1: Xtest1, x2: Xtest2, y: ytest})
        print "Testing: IZY={:.2f}\tIZX1={:.2f}\tIZX2={:.2f}\tacc={:.4f}\tavg_acc={:.4f}\terr={:.4f}\tavg_err={:.4f}".format(IZY, IZX1, IZX2, acc, avg_acc, 1-acc, 1-avg_acc)
        test_acc.append(acc)
        test_avg_acc.append(avg_acc)
        test_error.append(1-acc)
        test_avg_error.append(1-avg_acc)
        test_IZY.append(IZY)
        test_IZX1.append(IZX1)
        test_IZX2.append(IZX2)
        sess.close()
    result_file.write("train: %s\ntrain_avg: %s\ntrain_err: %s\ntrain_avg_err: %s\ntrain_IZY: %s\ntrain_IZX1: %s\ntrain_IZX2: %s\ntest: %s\n test_avg: %s\ntest_err: %s\ntest_avg_err: %s\ntest_IZY: %s\ntest_IZX1: %s\ntest_IZX2: %s"%(train_acc, train_avg_acc, train_error, train_avg_error, train_IZY,train_IZX1, train_IZX2, test_acc, test_avg_acc, test_error, test_avg_error, test_IZY, test_IZX1, test_IZX2))
    result_file.close()
