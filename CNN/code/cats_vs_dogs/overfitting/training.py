import os
import numpy as np
import tensorflow as tf
import input_data
import model_add_layers as model
import shutil

N_CLASSES = 2
IMG_W = 227
IMG_H = 227
BATCH_SIZE = 128
CAPACITY = 2000
MAX_STEP = 10000
learning_rate = 1e-4

# def run_training():
print 'in run_training'
train_dir = './data/train/'
logs_train_dir = './logs/train'
logs_validation_dir = './logs/validation'

def train():
    train, train_label, validation, validation_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY,
                                                          True)
    validation_batch, validation_label_batch = input_data.get_batch(validation,
                                                                    validation_label,
                                                                    IMG_W,
                                                                    IMG_H,
                                                                    BATCH_SIZE,
                                                                    CAPACITY,
                                                                    False)

    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape = [BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32)
    
    logits = model.inference(x, BATCH_SIZE, N_CLASSES, keep_prob)
    loss = model.losses(logits, y_)
    train_op = model.training(loss, learning_rate)
    acc = model.evaluation(logits, y_)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    validation_writer = tf.summary.FileWriter(logs_validation_dir, sess.graph)
    print 'before sess.run'
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([train_batch, train_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, acc],
                                            feed_dict = {x: tra_images, y_: tra_labels, keep_prob:
                                                         0.5})
            
            if step % 50 == 0:
                print('Step %d, train loss= %.2f  train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op, feed_dict = {x: tra_images, y_: tra_labels,
                                                                keep_prob: 0.5})
                train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_iamges, val_labels = sess.run([validation_batch, validation_label_batch])
                val_loss, val_acc = sess.run([loss, acc],
                                             feed_dict = {x: val_iamges, y_: val_labels, keep_prob:
                                                         1.0})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%% **' % (step, val_loss,
                                                                                  val_acc * 100.0))
                summary_str = sess.run(summary_op, feed_dict = {x: tra_images, y_: tra_labels,
                                                                keep_prob: 1.0})
                validation_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step = step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

def main():
    if not os.path.exists(train_dir):
        print "please create train data dir: ./data/train, and download data to this dir"
    if not os.path.exists(logs_train_dir):
        os.makedirs(logs_train_dir)
    else:
        shutil.rmtree(logs_train_dir)

    if not os.path.exists(logs_validation_dir):
        os.makedirs(logs_validation_dir)
    else:
        shutil.rmtree(logs_validation_dir)

    train()


if __name__ == "__main__":
    main()
