import tensorflow as tf
import pytoolkit.tensorflow as tl

class cifar10_net:
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 512, 512, 3), name='input_image')
        self.ph_label = tf.placeholder(tf.float32, shape=(self.batch_size), name='input_label')

        self.bn = tl.batch_norm(name='bn')

        self.logits = self.inference(self.ph_image)
        self.loss = self.compute_loss(self.ph_label, self.logits)
        self.acc = self.compute_acc(self.ph_label, self.logits)

        self.optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def _conv_layer(self, input, out_channels, conv_ksize, conv_stride_size, pool_ksize, pool_stride_size, name):
        with tf.variable_scope(name) as scope:
            lconv = tl.conv2d(input, out_channels, conv_ksize, conv_stride_size, name='conv')
            lpool = tl.max_pool2d(lconv, pool_ksize, pool_stride_size, name='pool')
            lbn = tl.bn_new(lpool)
            lrelu = tl.relu(lbn)
            return lrelu

    def inference(self, im):
        l0 = im
        l1 = self._conv_layer(l0, 32, (5,5), (1,1), (2,2), (2,2), name='l1')
        l2 = self._conv_layer(l1, 32, (5,5), (1,1), (2,2), (2,2), name='l2')
        l3 = self._conv_layer(l2, 64, (5,5), (1,1), (2,2), (2,2), name='l3')
        fc0 = tf.reshape(l3, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 64, name='fc1')
        fc2 = tl.fc(fc1, 101, name='fc2')
        return fc2

    def compute_loss(self, labels, logits):
        labels = tf.cast(labels, tf.int32)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
        xentropy = tf.reduce_mean(xentropy, name='xentropy_mean')
        return xentropy

    def compute_acc(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        acc = tf.equal(tf.argmax(logits, 1), labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc




