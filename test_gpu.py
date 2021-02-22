import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# 图像大小
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MAX_CAPTCHA = 4
CHAR_SET_LEN = 10

input = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])


# 定义CNN
def crack_captcha_cnn(x=input, w_alpha=0.01, b_alpha=0.1):
    # conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv1, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


# 加载网络
evaluate_net = crack_captcha_cnn()

with tf.Session() as sess:
    # 网络结构写入
    summary_writer = tf.summary.FileWriter('./log/', sess.graph)
    # summary_writer = tf.summary.FileWriter('./log/', tf.get_default_graph())

print('OK')