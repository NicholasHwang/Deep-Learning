import tensorflow as tf

def weight_variable(name, shape, initializer_stddev, wd):
    var = tf.get_variable(name,
                           shape = shape,
                           dtype = tf.float32,
                           initializer = 
                           tf.truncated_normal_initializer(stddev = initializer_stddev, dtype = tf.float32))
    #L2 normalization
#     if wd:
        # weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        # tf.add_to_collection('losses', weight_decay)
    return var

def conv_operation(data, weights, biases, strides, padding, name):
    conv = tf.nn.conv2d(data, weights, strides, padding)
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name)
    return conv1

def biases_variable(name, shape, initializer_const):
    return tf.get_variable(name,
                           shape = shape,
                           dtype = tf.float32,
                           initializer = tf.constant_initializer(initializer_const))

def max_pool_3x3(data, name):
    return tf.nn.max_pool(data, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
                               padding = 'SAME', name = name)

def lrn(data, name):
    return tf.nn.lrn(data, depth_radius = 4, bias = 1.0, alpha = 0.001 / 9.0,
                          beta = 0.75, name = name)

def inference(images, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = weight_variable('weights', [3, 3, 3, 32], 0.1, 0.0)
        biases = biases_variable('biases', [32], 0.1)
        conv1 = conv_operation(images, weights, biases, [1, 1, 1, 1], 'SAME', scope.name)
    
    with tf.variable_scope('pooling1_lrn') as scope:
        norm1 = lrn(conv1, 'norm1')
        pool1 = max_pool_3x3(norm1, 'pooling1')

    with tf.variable_scope('conv2') as scope:
        weights = weight_variable('weights', [3, 3, 32, 64], 0.1, 0.0)
        biases = biases_variable('biases', [64], 0.1)
        conv2 = conv_operation(pool1, weights, biases, [1, 1, 1, 1], 'SAME', scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = lrn(conv2, 'norm2')
        pool2 = max_pool_3x3(norm2, 'pooling2')

    with tf.variable_scope('conv3') as scope:
        weights = weight_variable('weights', [3, 3, 64, 128], 0.1, 0.0)
        biases = biases_variable('biases', [128], 0.1)
        conv3 = conv_operation(pool2, weights, biases, [1, 1, 1, 1], 'SAME', scope.name)

    with tf.variable_scope('conv4') as scope:
        weights = weight_variable('weights', [3, 3, 128, 64], 0.1, 0.0)
        biases = biases_variable('biases', [64], 0.1)
        conv4 = conv_operation(conv3, weights, biases, [1, 1, 1, 1], 'SAME', scope.name)

    with tf.variable_scope('conv5') as scope:
        weights = weight_variable('weights', [3, 3, 64, 32], 0.1, 0.0)
        biases = biases_variable('biases', [32], 0.1)
        conv5 = conv_operation(conv4, weights, biases, [1, 1, 1, 1], 'SAME', scope.name)

    with tf.variable_scope('pooling3'):
        pool3 = max_pool_3x3(conv5, 'pooling3')

    with tf.variable_scope('local1') as scope:
        reshape = tf.reshape(pool3, shape = [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = weight_variable('weights', [dim, 128], 0.005, 0.004)
        biases = biases_variable('biases', [128], 0.1)
        local1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)
        dropout1 = tf.nn.dropout(local1, 0.5)

    with tf.variable_scope('local2') as scope:
        weights = weight_variable('weights', [128, 128], 0.005, 0.004)
        biases = biases_variable('biases', [128], 0.1)
        local2 = tf.nn.relu(tf.matmul(dropout1, weights) + biases, name = scope.name)
        dropout1 = tf.nn.dropout(local2, 0.5)

    with tf.variable_scope('softmax_linear') as scope:
        weights = weight_variable('softmax_linear', [128, n_classes], 0.005, 0.0)
        biases = biases_variable('biases', [n_classes], 0.1)
        softmax_linear = tf.add(tf.matmul(local2, weights), biases, name = scope.name)

    return softmax_linear

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits = logits, labels = labels, name = 'xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name = 'loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        train_op = optimizer.minimize(loss, global_step = global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
