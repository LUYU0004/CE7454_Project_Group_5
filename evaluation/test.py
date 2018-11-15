# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 100
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


# you need to change the directories to yours.
bs_dir = '/home/fnzhan/datasets/cartoon/train/'
# train_dir = bs_dir + '/train/'
logs_train_dir = bs_dir + 'logs/'

# train, train_label = input_data.get_files()
#
# train_batch, train_label_batch = input_data.get_batch(train,
#                                                       train_label,
#                                                       IMG_W,
#                                                       IMG_H,
#                                                       BATCH_SIZE,
#                                                       CAPACITY)
# train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
# train_loss = model.losses(train_logits, train_label_batch)
# train_op = model.trainning(train_loss, learning_rate)
# train__acc = model.evaluation(train_logits, train_label_batch)
#
# summary_op = tf.summary.merge_all()
# sess = tf.Session()
# train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
# saver = tf.train.Saver()
#
# sess.run(tf.global_variables_initializer())
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)

bs_dir = '/home/fnzhan/datasets/cartoon/train/'
fake_sets = ['basic/', 'content_l1/', 'starGAN/', 'style/', 'style_vggface/']

fake_dir = bs_dir + 'real_cartoon/'
real_dir = bs_dir + 'iiit-cfw/'

def get_files():
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    # for set in fake_sets:
    # fake_dir = bs_dir + set

    fake_nms = os.listdir(fake_dir)
    fake_nms = fake_nms[2500:5000]
    for nm in fake_nms:
        fake_path = fake_dir + nm
        cats.append(fake_path)
        label_cats.append(0)

    real_nms = os.listdir(real_dir)
    real_nms = real_nms[2500:2525]
    for nm in real_nms:
        real_path = real_dir + nm
        dogs.append(real_path)
        label_dogs.append(1)
    print('There are %d cats\nThere are %d dogs' % (len(cats), len(dogs)))

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    label_list = [int(i) for i in label_list]

    return image_list, label_list

# def evaluate_all_image():
    # test_dir = '/home/fnzhan/datasets/cartoon/train/'
N_CLASSES = 2
print('-------------------------')
test, test_label = get_files()
# BATCH_SIZE = len(test)
print('There are %d test images totally..' % BATCH_SIZE)
print('-------------------------')
test_batch, test_label_batch = input_data.get_batch(test,
                                                    test_label,
                                                    IMG_W,
                                                    IMG_H,
                                                    BATCH_SIZE,
                                                    CAPACITY)

print (test_batch.shape)

logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
testloss = model.losses(logits, test_label_batch)
testcorr = model.test_evaluation(logits, test_label_batch)
print (testcorr.shape)

logs_train_dir = bs_dir + 'logs/'
saver = tf.train.Saver()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
with tf.Session() as sess:
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    print('-------------------------')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for i in range(25):
    if 1:
        test_lab, test_corr = sess.run([test_label_batch, testcorr])
        # print (test_lab)
        print (test_corr)

    # print('The model\'s loss is %.2f' % test_loss)
    # correct = int(BATCH_SIZE * test_corr)
    # print (correct)
    # print('Correct : %d' % correct)
    # print('Wrong : %d' % (BATCH_SIZE - correct))
    # print('The accuracy in test images are %.2f%%' % (test_corr * 100.0))
coord.request_stop()
coord.join(threads)
sess.close()