import tensorflow as tf
import numpy as np
import input_data
import model_add_layers as model
import matplotlib.pyplot as plt

N_CLASSES = 2
BATCH_SIZE = 10
IMG_W = 208
IMG_H = 208
CAPACITY = 2000

test_dir = './data/test/'
test, test_index = input_data.get_test_files(test_dir)
test_len = len(test_index)
test_batch, test_index_list = input_data.get_test_batch(test,
                                                        test_index,
                                                    IMG_W,
                                                    IMG_H,
                                                    BATCH_SIZE,
                                                    CAPACITY)

# x = tf.placeholder(tf.float32, shape = [1, IMG_W, IMG_H, 3])
# train_dir = './data/train/'
# train_image, train_label, _, _ = input_data.get_files(train_dir)
# train_len = len(train_label)
# train_image_batch, train_label_batch = input_data.get_batch(train_image,
                                                            # train_label,
                                                            # IMG_W,
                                                            # IMG_H,
                                                            # BATCH_SIZE,
                                                            # CAPACITY,
                                                            # False)


test_logit = model.inference(test_batch, BATCH_SIZE, N_CLASSES, 1)
test_logit = tf.nn.softmax(test_logit)
# test_logit = tf.reduce_mean(test_logit, name = 'loss')

logs_train_dir = './logs/train/'
saver = tf.train.Saver()

sess = tf.Session()

print 'Reading checkpoints...'
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print 'Loading success, global step is %s' % global_step
else:
    print 'No checkpoint file found'

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)
try:
    i = 0
    while i < test_len:
    # for i in np.arange(test_len):
        prediction = sess.run(test_logit)
        # logit = model.inference(img, BATCH_SIZE, N_CLASSES)
        # prediction = tf.nn.softmax(logit)
        # print 'step:%d index:%d class:%d' % (i, index_list, np.argmax(prediction))
        j = 0
        while j < BATCH_SIZE:
            print 'step:%d index:%d class:%d' % (i+j, test_index_list[j].eval(session=sess), np.argmax(prediction[j]))
            j+=1
        i+=10
#     while not coord.should_stop():
        # img, index = sess.run([test_batch, test_index_list])
        # for j in np.arange(BATCH_SIZE):
                # plt.imshow(img[j,:,:,:])
                # plt.show()
                # # logit = model.inference(img[j,:,:,:], 1, N_CLASSES)
                # # prediction = tf.nn.softmax(logit)
                # prediction = sess.run(test_logit,feed_dict={x: img})
                # print('label: %d prediction:%d' %(index[j], np.argmax(prediction)))
except tf.errors.OutOfRangeError:
    print('done!')
finally:
    coord.request_stop()

coord.join(threads)
print 'after prediction'
sess.close()
