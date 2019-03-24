import tensorflow as tf
import nn_layers
import preprocess_images as procIm
import os
import numpy as np
from sklearn.utils import shuffle

imSize = 64
cropSize = 60
numChan = 2
dequeueSize = 100
decay_step = 25
decay_rate = 0.96
stretchLow = 0.1  # stretch channels lower percentile
stretchHigh = 99.9  # stretch channels upper percentile
SAVE_INTERVAL = 500


class cnn_model:
    def __init__(self, epochs, batchsize, num_classes, learning_rate, is_transfer, checkpoint_dir=None,
                 save_new_checkpoint_dir=None):
        # value the params(batchsize, steps, checkpoint, learning_rate)
        # define model and define loss and optimizer
        # hold the saver about pre-trained model(self.checkpoint_saver and self.final_saver)
        # used self.transfer to determine use the checkpoint_saver or final_saver to restore the checkpoint
        self.epochs = epochs
        self.batchsize = batchsize
        self.num_classes = num_classes
        self.is_transfer = is_transfer
        self.checkpoint_dir = checkpoint_dir
        self.save_new_checkpoint_dir = save_new_checkpoint_dir

        self.global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate, self.global_step * dequeueSize,
                                                   decay_step * dequeueSize, decay_rate, staircase=True)

        # define the model
        self.is_training = tf.placeholder(tf.bool, [], name='is_training')  # for batch normalization
        self.input = tf.placeholder('float32', shape=[None, cropSize, cropSize, numChan], name='input')
        self.labels = tf.placeholder('float32', shape=[None, self.num_classes], name='labels')
        self.keep_prob = tf.placeholder(tf.float32)
        self.logits = self.__DeepLocModel(self.input, self.is_training)
        self.predicted_y = tf.nn.softmax(self.logits, name='softmax')

        self.variables2restore = []
        self.newVariables = []
        for x in tf.global_variables():
            if 'final_layer' in x.name or 'pred_layer' in x.name:
                self.newVariables.append(x)
            else:
                self.variables2restore.append(x)

        self.acc = self.__accuracy(self.predicted_y, self.labels)
        self.cross_entropy = self.__loss_logits(self.logits, self.labels)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy, global_step=self.global_step)

        # saver
        self.saver = tf.train.Saver(self.variables2restore)
        self.final_saver = tf.train.Saver()

    def train(self, train_images, train_labels, eval_images, eval_labels):
        # images should be [AllImgNum, features], labels should be [AllImgNum, numClasses]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={self.is_training: True})
            if self.checkpoint_dir is not None:
                print ('...doing transfer learning...')
                # get the newest checkpoint
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                    # transfer learning use saver to restore, retraining use final_saver
                    if self.is_transfer:
                        self.saver.restore(sess, os.path.join(self.checkpoint_dir, ckpt_name))
                    else:
                        self.final_saver.restore(sess, os.path.join(self.checkpoint_dir, ckpt_name))
                else:
                    print ('...no checkpoint found...')
                    return
            # start training
            batch = {}
            for epoch in range(self.epochs):
                train_images, train_labels = shuffle(train_images, train_labels)
                for index in range(0, train_images.shape[0], self.batchsize):
                    batch['data'] = train_images[index:index+self.batchsize]
                    batch['labels'] = train_labels[index:index+self.batchsize]
                    processedBatch = procIm.preProcessImages(batch['data'],
                                                imSize, cropSize, numChan,
                                                rescale=False, stretch=True,
                                                means=None, stds=None,
                                                stretchLow=stretchLow, stretchHigh=stretchHigh)

                    _, cur_train_acc, cur_train_loss = sess.run([self.train_step, self.acc, self.cross_entropy],
                                                                feed_dict={self.is_training: True,
                                                                           self.keep_prob: 0.5,
                                                                           self.input: processedBatch,
                                                                           self.labels: batch['labels']})
                    print('Train accuracy at epoch %s batch %s: %s, loss: %s' % (epoch+1, index/self.batchsize, cur_train_acc, cur_train_loss))

                    numIter = epoch*(train_images.shape[0]/self.batchsize)+index/self.batchsize
                    if numIter % SAVE_INTERVAL == 0:
                        # save a new checkpoint in every SAVE_INTERVAL Iteration.
                        if self.save_new_checkpoint_dir is not None:
                            checkpoint_path = os.path.join(self.save_new_checkpoint_dir, 'model.ckpt')
                            self.final_saver.save(sess, checkpoint_path, global_step=numIter)
                        else:
                            print('you should save your new checkpoint file')
                            return
                    if numIter % 10 == 0:
                        # valid print every epoch
                        test_eval_images, test_eval_labels = shuffle(eval_images, eval_labels)
                        test_eval_images = test_eval_images[:self.batchsize]
                        test_eval_labels = test_eval_labels[:self.batchsize]
                        eval_imgs = procIm.preProcessImages(test_eval_images,
                                                        imSize, cropSize, numChan,
                                                        rescale=False, stretch=True,
                                                        means=None, stds=None,
                                                        stretchLow=stretchLow, stretchHigh=stretchHigh,
                                                        jitter=True, randTransform=False)

                        eval_acc = sess.run([self.acc], feed_dict={self.is_training: False,
                                                                self.keep_prob: 1.0,
                                                                self.input: eval_imgs,
                                                                self.labels: test_eval_labels})
                        print ('For all the testing data, you get the accuracy is %s' % eval_acc)
                    if numIter == 1000:
                        print('end training')
                        return

    def eval(self, images, labels):
        # images should be [AllImgNum, features], labels should be [AllImgNum, numClasses]
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                self.final_saver.restore(sess, os.path.join(self.checkpoint_dir, ckpt_name))
            else:
                print ('...no checkpoint found...')
                return
            # start evaluating
            images = procIm.preProcessImages(images,
                                             imSize, cropSize, numChan,
                                             rescale=False, stretch=True,
                                             means=None, stds=None,
                                             stretchLow=stretchLow, stretchHigh=stretchHigh,
                                             jitter=True, randTransform=False)

            eval_acc = sess.run([self.acc], feed_dict={self.is_training: False,
                                                       self.keep_prob: 1.0,
                                                       self.input: images,
                                                       self.labels: labels})
            print ('For all the testing data, you get the accuracy is %s' % eval_acc)

    def __DeepLocModel(self, input_images, is_training):
        conv1 = nn_layers.conv_layer(input_images, 3, 3, 2, 64, 1, 'conv_1', is_training=is_training)
        conv2 = nn_layers.conv_layer(conv1, 3, 3, 64, 64, 1, 'conv_2', is_training=is_training)
        pool1 = nn_layers.pool2_layer(conv2, 'pool1')
        conv3 = nn_layers.conv_layer(pool1, 3, 3, 64, 128, 1, 'conv_3', is_training=is_training)
        conv4 = nn_layers.conv_layer(conv3, 3, 3, 128, 128, 1, 'conv_4', is_training=is_training)
        pool2 = nn_layers.pool2_layer(conv4, 'pool2')
        conv5 = nn_layers.conv_layer(pool2, 3, 3, 128, 256, 1, 'conv_5', is_training=is_training)
        conv6 = nn_layers.conv_layer(conv5, 3, 3, 256, 256, 1, 'conv_6', is_training=is_training)
        conv7 = nn_layers.conv_layer(conv6, 3, 3, 256, 256, 1, 'conv_7', is_training=is_training)
        conv8 = nn_layers.conv_layer(conv7, 3, 3, 256, 256, 1, 'conv_8', is_training=is_training)
        pool3 = nn_layers.pool2_layer(conv8, 'pool3')
        pool3_flat = tf.reshape(pool3, [-1, 8 * 8 * 256])
        fc_1 = nn_layers.nn_layer(pool3_flat, 8 * 8 * 256, 512, 'fc_1', act=tf.nn.relu, is_training=is_training)
        fc_2 = nn_layers.nn_layer(fc_1, 512, 512, 'fc_2', act=tf.nn.relu, is_training=is_training)
        fc2_drop = tf.nn.dropout(fc_2, self.keep_prob)
        logit = nn_layers.nn_layer(fc2_drop, 512, self.num_classes, 'final_layer', act=None, is_training=is_training)

        return logit

    def __loss(self, predicted_y, labeled_y):
        with tf.name_scope('cross_entropy'):
            diff = labeled_y * tf.log(tf.clip_by_value(predicted_y, 1e-16, 1.0))
            with tf.name_scope('total'):
                cross_entropy = -tf.reduce_mean(diff)

        return cross_entropy

    def __accuracy(self, predicted_y, labeled_y):
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(labeled_y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def __loss_logits(self, logits, labeled_y):
        with tf.name_scope('cross_entropy'):
            logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labeled_y,
                                                                      name='sigmoid_cross_entropy')
            cross_entropy = tf.reduce_mean(logistic_losses)
            tf.summary.scalar('cross entropy', cross_entropy)

        return cross_entropy

    def __loss_numpy(self, y_pred, y_lab):
        cross_entropy = -np.mean(y_lab * np.log(np.clip(y_pred, 1e-16, 1.0)))
        return cross_entropy

    def __accuracy_numpy(self, y_pred, y_lab):
        accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(y_lab, 1))
        return accuracy
