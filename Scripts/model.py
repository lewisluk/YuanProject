# Author LU Yifu
import os
from scripts.network_ops import *
from tensorflow.contrib.layers import flatten
from scripts.metrics import metrics
from scripts.data_io import io_handler

class ResNext():
    def __init__(self, epochs=100, batchsize=4, learning_rate=1e-3, blocks=2, cardinality=4, depth=32):
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 235, 1], name='x_input')
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_label')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.result_path = 'result'
        self.data_path = 'strawberries'
        self.class0_dir_name = 'negatives'
        self.class1_dir_name = 'positives'
        self.batchsize = batchsize
        self.mylayers = layers(depth, cardinality, blocks)

    def build(self, inputs):
        input_x = self.mylayers.first_layer(inputs, scope='first_layer')

        x = self.mylayers.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.mylayers.residual_layer(x, out_dim=128, layer_num='2')
        x = self.mylayers.residual_layer(x, out_dim=256, layer_num='3')

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x

    def net_init(self):
        self.prediction = self.build(self.input)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction,
                                                            labels=self.label)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=self.epochs)
        self.sess.run(tf.global_variables_initializer())
        print('Network initialized')

        self.my_io_handler = io_handler(self.batchsize,
                                   self.data_path,
                                   self.class0_dir_name,
                                   self.class1_dir_name)
        self.X_train, self.X_test, self.y_train, self.y_test = self.my_io_handler.get_train_val_names()
        print('Dataset divided')

    def start_training(self, load_model, index):
        if load_model:
            self.saver.restore(self.sess, os.path.join(self.result_path,
                                                       'model_{:03d}.ckpt'.format(index)))
        for i in range(self.epochs):
            # initialize metrics for training data evaluation
            my_metrics = metrics(self.batchsize)
            train_loss = []
            # calculate how many iterations per epoch for training
            iters, mod = divmod(len(self.X_train), self.batchsize)
            # start training loop
            for j in range(iters):
                input_batch, label_batch \
                    = self.my_io_handler.load_image_label_batch(j, self.X_train, self.y_train)
                train_dict = {self.input:input_batch,
                              self.label:label_batch}
                _, loss, predict = self.sess.run([self.opt, self.loss, self.prediction],
                                        feed_dict=train_dict)
                my_metrics.accumulate(predict, label_batch)
                train_loss.append(loss)

            # get training metrics from my_metrics
            acc, prec0, recall0, prec1, recall1, mAP = \
            my_metrics.get_acc(),\
            my_metrics.get_precision(0),\
            my_metrics.get_recall(0),\
            my_metrics.get_precision(1),\
            my_metrics.get_recall(1),\
            my_metrics.get_mAP()

            print('Train\t\tEp:{:d}, loss:{:.4f}, acc:{:.4f}, prec0:{:.4f}, '
                  'recall0:{:.4f}, prec1:{:.4f}, recall1:{:.4f}, mAP:{:.4f}'.format(
                i+1, np.mean(train_loss), acc, prec0, recall0, prec1, recall1, mAP
            ))

            # initialize metrics for validation data evaluation
            my_metrics = metrics(self.batchsize)
            val_loss = []
            # calculate how many iterations per epoch for validation
            iters, mod = divmod(len(self.X_test), self.batchsize)
            # start validation loop
            for j in range(iters):
                input_batch, label_batch \
                    = self.my_io_handler.load_image_label_batch(j, self.X_test, self.y_test)
                val_dict = {self.input:input_batch,
                              self.label:label_batch}
                loss, predict = self.sess.run([self.loss, self.prediction],
                                        feed_dict=val_dict)
                my_metrics.accumulate(predict, label_batch)
                val_loss.append(loss)

            # get validation metrics from my_metrics
            acc, prec0, recall0, prec1, recall1, mAP = \
                my_metrics.get_acc(), \
                my_metrics.get_precision(0), \
                my_metrics.get_recall(0), \
                my_metrics.get_precision(1), \
                my_metrics.get_recall(1), \
                my_metrics.get_mAP()
            print('Validation \tEp:{:d}, loss:{:.4f}, acc:{:.4f}, prec0:{:.4f}, '
                  'recall0:{:.4f}, prec1:{:.4f}, recall1:{:.4f}, mAP:{:.4f}'.format(
                i + 1, np.mean(val_loss), acc, prec0, recall0, prec1, recall1, mAP
            ))
            self.saver.save(self.sess, 'result/model_{:03d}.ckpt'.format(i + 1))
