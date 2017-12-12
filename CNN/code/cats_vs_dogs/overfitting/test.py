import tensorflow as tf
import numpy as np
import input_data
import model_add_layers as model

N_CLASSES = 2
BATCH_SIZE = 1
IMG_W = 208
IMG_H = 208
CAPACTIY = 2000

test_dir = './data/test/'
test, test_index = input_data.get_test_files(test_dir)
test_len = len(test_index)
test_batch = input_data.get_test_batch(test,
                                                    IMG_W,
                                                    IMG_H,
                                                    BATCH_SIZE,
                                                    CAPACTIY)

# x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMG_W, IMG_H, 3])
test_logit = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
test_logit = tf.nn.softmax(test_logit)

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
for i in np.arange(test_len):
    prediction = sess.run(test_logit)
    print 'step:%d index:%d class:%d' % (i, int(test_index[i]), np.argmax(prediction))

coord.request_stop()
coord.join()
print 'after prediction'
sess.close()
