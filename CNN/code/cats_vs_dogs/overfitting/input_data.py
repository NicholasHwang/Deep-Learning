import tensorflow as tf
import numpy as np
import os
import math

ratio = 0.3

def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    print('%d cats \n%d dogs'%(len(cats), len(dogs)))
    #the following codes only to do just once shuffle
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
 
    temp = np.array([image_list, label_list])#shape(1,2)
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    n_sample = len(label_list)
    n_validation_value = math.ceil(n_sample * ratio)
    n_train_value = int(n_sample - n_validation_value)
    print('sample:%d validation value:%d train value:%d' % (n_sample, n_validation_value, n_train_value))
    train_image_list = image_list[0: n_train_value]
    train_label_list = label_list[0: n_train_value]
    # train_label_list = [int(i) for i in train_label_list]
    validation_image_list = image_list[n_train_value: -1]
    validation_label_list = label_list[n_train_value: -1]
    # validation_label_list = [int(i) for i in validation_label_list]
    
    return train_image_list, train_label_list, validation_image_list, validation_label_list

def get_batch(image, label, image_w, image_h, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels = 3)
    #data augmentation
#     image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta = 63)
    # image = tf.image.random_contrast(image, lower = 0.2, upper = 1.8)
 
    image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    # image = tf.image.per_image_standardization(image)
    # image = tf.image.resize_images(image, [image_h, image_w], method =
                                   # tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)

    image_batch, label_batch = tf.train.batch([image,label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = capacity)

    # image_batch = tf.cast(image_batch, tf.float32)
    # label_batch = tf.reshape(label_batch, [batch_size])
    # print image_batch.get_shape()
    return image_batch, label_batch

def get_test_files(test_dir):
    animals = []
    indexs = []
    for file in os.listdir(test_dir):
        name = file.split('.')
        animals.append(test_dir + file)
        indexs.append(name[0])

    index = [int(i) for i in indexs]
    return  animals, indexs

def get_test_batch(image, image_w, image_h, batch_size, capacity):
    image = tf.cast(image, tf.string) #image list has only file names
    input_queue = tf.train.slice_input_producer([image])
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels = 3)
    image = tf.image.resize_images(image, [image_h, image_w], method = 
                                   tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    image_batch = tf.train.batch([image],
                                 batch_size = batch_size,
                                 num_threads = 64,
                                 capacity = capacity)
    return image_batch

#for test

# if __name__ == "__main__":
# import matplotlib.pyplot as plt
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208

# train_dir = './data/train/'
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# with tf.Session() as sess:
    # i = 0
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord = coord)
    # try:
        # while not coord.should_stop() and i < 1:
            # img, label = sess.run([image_batch, label_batch])
            # for j in np.arange(BATCH_SIZE):
                # print('label: %d' %label[j])
                # plt.imshow(img[j,:,:,:])
                # plt.show()
            # i += 1
    # except tf.errors.OutOfRangeError:
        # print('done!')
    # finally:
        # coord.request_stop()
    # coord.join(threads)
    # sess.close()
