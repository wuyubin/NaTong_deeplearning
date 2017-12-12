import tensorflow as tf
import pytoolkit.tensorflowOwn as tl

class cifar10_net:
    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        if FLAGS.net_type is 'AlexNet':
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 277, 277, 3), name='input_image')
        elif FLAGS.net_type is 'VGG16' or  'VGG19':
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 224, 224, 3), name='input_image')
        elif FLAGS.net_type is 'ResNet':
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 224, 224, 3), name='input_image')
        elif FLAGS.net_type is 'Inception':
            # self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 227, 227, 3), name='input_image')
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 512, 512, 3), name='input_image')
        elif FLAGS.net_type is 'MobileNet':
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 224, 224, 3), name='input_image')
        elif FLAGS.net_type is 'Basic':
            self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3), name='input_image')

        self.ph_label = tf.placeholder(tf.float32, shape=(self.batch_size), name='input_label')
        self.bn = tl.batch_norm(name='bn')
        self.keep_prob = tf.placeholder(tf.float32, name='drop_out')
        self.net_type = FLAGS.net_type
        if self.net_type is 'AlexNet':
            self.logits = self.inference_AlexNet(self.ph_image)
        elif self.net_type is 'VGG16':
            self.logits = self.inference_VGG16(self.ph_image)
        elif self.net_type is 'VGG19':
            self.logits = self.inference_VGG19(self.ph_image)
        elif self.net_type is 'ResNet':
            self.logits = self.inference_ResNet(self.ph_image)
        elif FLAGS.net_type is 'Inception':
            self.logits = self.inference_Inceptionv3(self.ph_image)
        elif FLAGS.net_type is 'MobileNet':
            self.logits = self.inference_MobileNet(self.ph_image)
        elif FLAGS.net_type is 'Basic':
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
        l1 = self._conv_layer(l0, 32, (5, 5), (1, 1), (2, 2), (2, 2), name='l1')
        l2 = self._conv_layer(l1, 32, (5, 5), (1, 1), (2, 2), (2, 2), name='l2')
        l3 = self._conv_layer(l2, 64, (5, 5), (1, 1), (2, 2), (2, 2), name='l3')
        fc0 = tf.reshape(l3, [self.batch_size, -1])
        fc1 = tl.fc(fc0, 64, name='fc1')
        fc2 = tl.fc(fc1, 10, name='fc2')
        return fc2
    def _AlexNet_conv_layer(self, input, out_channels, conv_ksize, conv_stride_size, pool_ksize, pool_stride_size, name):
        with tf.variable_scope(name) as scope:
            lconv = tl.conv2d(input, out_channels, conv_ksize, conv_stride_size, name='conv')
            lrelu = tl.relu(lconv)
            lnorm = tl.lrn(lrelu, depth_radius=2, bias=1.0, alpha=2e-05, beta=0.75)
            lpool = tl.max_pool2d(lnorm, pool_ksize, pool_stride_size,  padding='VALID', name='pool')
            return lpool
			
    def inference_AlexNet(self,im):
        l1 = self._AlexNet_conv_layer(im, 96, (11,11), (4,4), (3,3), (2,2), name='l1')
        l2 = self._AlexNet_conv_layer(l1, 256, (5,5), (1,1), (3,3), (2,2), name='l2')
        with tf.variable_scope('l3') as scope:
            l3_conv = tl.conv2d(l2, 384, (3,3), (1,1), name='l3')
            l3 = tl.relu(l3_conv)
        with tf.variable_scope('l4') as scope:
            l4_conv = tl.conv2d(l3, 384, (3,3), (1,1), name='l4')
            l4 = tl.relu(l4_conv)
        with tf.variable_scope('l5') as scope:
            l5_lconv = tl.conv2d(l4, 256, (3,3), (1,1), name='l5')
            l5_lrelu = tl.relu(l5_lconv)
            l5 = tl.max_pool2d(l5_lrelu, (3,3), (2,2),  padding='VALID', name='pool')
        #print l5.get_shape()
        fc0 = tf.reshape(l5, [self.batch_size, -1])

        fc1 = tl.fc(fc0, 4096, name='fc1')

        l6_lrelu = tl.relu(fc1)
        l6 = tl.dropout(l6_lrelu, self.keep_prob)
        fc2 = tl.fc(l6, 4096, name='fc2')
        l7_lrelu = tl.relu(fc2)
        l7 = tl.dropout(l7_lrelu, self.keep_prob)
        fc3 = tl.fc(l7, 10, name='fc3')
        return fc3

    def inference_VGG16(self, im):
        with tf.variable_scope('l1_conv1') as scope:
            l1_conv1 = tl.conv2d(im, 64, (3,3), (1,1), name='l1')
        with tf.variable_scope('l2_conv2') as scope:
            l2_conv2 = tl.conv2d(l1_conv1, 64, (3, 3), (1, 1), name='l2')
            l2_pooling = tl.max_pool2d(l2_conv2, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l3_conv1') as scope:
            l3_conv1 = tl.conv2d(l2_pooling, 128, (3, 3), (1, 1), name='l3')
        with tf.variable_scope('l4_conv2') as scope:
            l4_conv2 = tl.conv2d(l3_conv1, 128, (3, 3), (1, 1), name='l4')
            l4_pooling = tl.max_pool2d(l4_conv2, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l5_conv1') as scope:
            l5_conv1 = tl.conv2d(l4_pooling, 256, (3, 3), (1, 1), name='l5')
        with tf.variable_scope('l6_conv2') as scope:
            l6_conv2 = tl.conv2d(l5_conv1, 256, (3, 3), (1, 1), name='l6')
        with tf.variable_scope('l7_conv3') as scope:
            l7_conv3 = tl.conv2d(l6_conv2, 256, (3, 3), (1, 1), name='l7')
            l7_pooling = tl.max_pool2d(l7_conv3, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l8_conv1') as scope:
            l8_conv1 = tl.conv2d(l7_pooling, 512, (3, 3), (1, 1), name='l8')
        with tf.variable_scope('l9_conv2') as scope:
            l9_conv2 = tl.conv2d(l8_conv1, 512, (3, 3), (1, 1), name='l9')
        with tf.variable_scope('l10_conv3') as scope:
            l10_conv3 = tl.conv2d(l9_conv2, 512, (3, 3), (1, 1), name='l10')
            l10_pooling = tl.max_pool2d(l10_conv3, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l11_conv1') as scope:
            l11_conv1 = tl.conv2d(l10_pooling, 512, (3, 3), (1, 1), name='l11')
        with tf.variable_scope('l12_conv2') as scope:
            l12_conv2 = tl.conv2d(l11_conv1, 512, (3, 3), (1, 1), name='l12')
        with tf.variable_scope('l13_conv3') as scope:
            l13_conv3 = tl.conv2d(l12_conv2, 512, (3, 3), (1, 1), name='l13')
            l13_pooling = tl.max_pool2d(l13_conv3, (3, 3), (2, 2), padding='SAME', name='pool')
            #shp = l13_pooling.get_shape()
            fc0 = tf.reshape(l13_pooling, [self.batch_size, -1])
            #fc0 = tf.reshape(l13_pooling, [shp[1].value * shp[2].value * shp[3].value, -1], name='resh1')

            fc1 = tl.fc(fc0, 4096, name='fc1')

            fc1_out = tl.dropout(fc1, self.keep_prob)
            fc2 = tl.fc(fc1_out, 4096, name='fc2')

            fc2_out = tl.dropout(fc2, self.keep_prob)
            fc3 = tl.fc(fc2_out, 10, name='fc3')
            print (fc3.get_shape())
            return fc3

    def inference_VGG19(self, im):
        with tf.variable_scope('l1_conv1') as scope:
            l1_conv1 = tl.conv2d(im, 64, (3, 3), (1, 1), name='l1')
        with tf.variable_scope('l2_conv2') as scope:
            l2_conv2 = tl.conv2d(l1_conv1, 64, (3, 3), (1, 1), name='l2')
            l2_pooling = tl.max_pool2d(l2_conv2, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l3_conv1') as scope:
            l3_conv1 = tl.conv2d(l2_pooling, 128, (3, 3), (1, 1), name='l3')
        with tf.variable_scope('l4_conv2') as scope:
            l4_conv2 = tl.conv2d(l3_conv1, 128, (3, 3), (1, 1), name='l4')
            l4_pooling = tl.max_pool2d(l4_conv2, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l5_conv1') as scope:
            l5_conv1 = tl.conv2d(l4_pooling, 256, (3, 3), (1, 1), name='l5')
        with tf.variable_scope('l6_conv2') as scope:
            l6_conv2 = tl.conv2d(l5_conv1, 256, (3, 3), (1, 1), name='l6')
        with tf.variable_scope('l7_conv3') as scope:
            l7_conv3 = tl.conv2d(l6_conv2, 256, (3, 3), (1, 1), name='l7')
        with tf.variable_scope('l8_conv4') as scope:
            l8_conv4 = tl.conv2d(l7_conv3, 256, (3, 3), (1, 1), name='l8')
            l8_pooling = tl.max_pool2d(l8_conv4, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l9_conv1') as scope:
            l9_conv1 = tl.conv2d(l8_pooling, 512, (3, 3), (1, 1), name='l9')
        with tf.variable_scope('l10_conv2') as scope:
            l10_conv2 = tl.conv2d(l9_conv1, 512, (3, 3), (1, 1), name='l10')
        with tf.variable_scope('l11_conv3') as scope:
            l11_conv3 = tl.conv2d(l10_conv2, 512, (3, 3), (1, 1), name='l11')
        with tf.variable_scope('l12_conv4') as scope:
            l12_conv4 = tl.conv2d(l11_conv3, 512, (3, 3), (1, 1), name='l12')
            l12_pooling = tl.max_pool2d(l12_conv4, (3, 3), (2, 2), padding='SAME', name='pool')

        with tf.variable_scope('l13_conv1') as scope:
            l3_conv1 = tl.conv2d(l12_pooling, 512, (3, 3), (1, 1), name='l13')
        with tf.variable_scope('l14_conv2') as scope:
            l14_conv2 = tl.conv2d(l3_conv1, 512, (3, 3), (1, 1), name='l14')
        with tf.variable_scope('l15_conv3') as scope:
            l15_conv3 = tl.conv2d(l14_conv2, 512, (3, 3), (1, 1), name='l15')
        with tf.variable_scope('l16_conv4') as scope:
            l16_conv4 = tl.conv2d(l15_conv3, 512, (3, 3), (1, 1), name='l16')
            l16_pooling = tl.max_pool2d(l16_conv4, (3, 3), (2, 2), padding='SAME', name='pool')
            fc0 = tf.reshape(l16_pooling, [self.batch_size, -1])

            fc1 = tl.fc(fc0, 4096, name='fc1')

            fc1_out = tl.dropout(fc1, self.keep_prob)
            fc2 = tl.fc(fc1_out, 4096, name='fc2')
            fc2_out = tl.dropout(fc2, self.keep_prob
                                 )
            fc3 = tl.fc(fc2_out, 10, name='fc3')

            return fc3

    def residual_block(self,x, n_out, subsample, name):
        with tf.variable_scope(name+'conv_1')as scope:
            if subsample:
                y = tl.conv2d(x, n_out, (3,3), (2,2), padding='SAME',name=name+'conv_1')
                shortcut = tl.conv2d(x, n_out, (3,3), (2,2), padding='SAME',name=name +'shortcut')
            else:
                y = tl.conv2d(x, n_out, (3,3), (1,1), padding='SAME', name=name+'conv_1')
                shortcut = tl.identity(x)
            y = tl.bn_new(y)
            y = tl.relu(y)
        with tf.variable_scope(name+'conv_2')as scope:
            y = tl.conv2d(y, n_out, (3,3), (1,1), padding='SAME', name=name+'conv_2')
            y = tl.bn_new(y)
            y = y + shortcut
            y = tf.nn.relu(y, name='relu_2')
        return y

    def residual_group(self,x, n_out, n, first_subsample, name):
        with tf.variable_scope('conv_'+name) as scope:
            y = self.residual_block(x, n_out, first_subsample, name='block_1'+name)
            for i in range(n - 1):
                y = self.residual_block(y, n_out, False, name='block_%d' % (i + 2))
            return y

    def inference_ResNet(self, im):
        n=5
        n_class=10
        with tf.variable_scope('conv_0') as scope:
            y = tl.conv2d(im, 16, (3,3), (1,1), padding='SAME', name='conv_init')
            y = tl.bn_new(y)
            y = tl.relu(y)
        y = self.residual_group(y, 16, n, False,  name='group_1')
        y = self.residual_group(y, 32, n, True, name='group_2')
        y = self.residual_group(y, 64, n, True, name='group_3')
            #print(y.get_shape())
        #shape = y.get_shape()
        #y = tl.avg_pool2d(y, shape[1:3], (1, 1), padding='VALID', name='avg_pool')
        y = tf.reshape(y, [self.batch_size, -1])
        y = tl.fc(y, n_class, name='fc')

        return y

    def Inceptionv3_module1(self, net, kernel_size,name):
        with tf.variable_scope(name):
            with tf.variable_scope('branch1x1') as scope:

                branch1x1 = tl.conv2d(net, 64, (1, 1))
                branch1x1 = tl.bn_new(branch1x1)
                branch1x1 = tl.relu(branch1x1)
            with tf.variable_scope('branch5x5') as scope:

                branch5x5 = tl.conv2d(net, 48, (1, 1),name='branch5x5_1')
                branch5x5 = tl.bn_new(branch5x5,name='branch5x5_1')
                branch5x5 = tl.relu(branch5x5)
                branch5x5 = tl.conv2d(branch5x5, 64, kernel_size,name='branch5x5_2')
                branch5x5 = tl.bn_new(branch5x5,name='branch5x5_2')
                branch5x5 = tl.relu(branch5x5)
            with tf.variable_scope('branch3x3dbl') as scope:

                branch3x3dbl = tl.conv2d(net, 64, (1, 1),name='branch3x3dbl_1')
                branch3x3dbl = tl.bn_new(branch3x3dbl,name='branch3x3dbl_1')
                branch3x3dbl = tl.relu(branch3x3dbl)
                branch3x3dbl = tl.conv2d(branch3x3dbl, 96, (3, 3),name='branch3x3dbl_2')
                branch3x3dbl = tl.bn_new(branch3x3dbl,name='branch3x3dbl_2')
                branch3x3dbl = tl.relu(branch3x3dbl)
                branch3x3dbl = tl.conv2d(branch3x3dbl, 96, (3, 3),name='branch3x3dbl_3')
                branch3x3dbl = tl.bn_new(branch3x3dbl,name='branch3x3dbl_3')
                branch3x3dbl = tl.relu(branch3x3dbl)
            with tf.variable_scope('branch_pool') as scope:

                branch_pool = tl.avg_pool2d(net, (3, 3))
                branch_pool = tl.conv2d(branch_pool, 32, (1, 1))
                branch_pool = tl.bn_new(branch_pool)
                branch_pool = tl.relu(branch_pool)
                net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool],3)

        return net

    def Inceptionv3_module2(self, net,name):
        with tf.variable_scope(name) as scope:

            with tf.variable_scope('branch1x1'):
                branch1x1 = tl.conv2d(net, 192, (1, 1))
                branch1x1 = tl.bn_new(branch1x1)
                branch1x1 = tl.relu(branch1x1)
            with tf.variable_scope('branch7x7'):
                branch7x7 = tl.conv2d(net, 192, (1, 1),name='branch7x7_1')
                branch7x7 = tl.bn_new(branch7x7,name='branch7x7_1')
                branch7x7 = tl.relu(branch7x7)
                branch7x7 = tl.conv2d(branch7x7, 192, (1, 7),name='branch7x7_2')
                branch7x7 = tl.bn_new(branch7x7,name='branch7x7_2')
                branch7x7 = tl.relu(branch7x7)
                branch7x7 = tl.conv2d(branch7x7, 192, (7, 1),name='branch7x7_3')
                branch7x7 = tl.bn_new(branch7x7,name='branch7x7_3')
                branch7x7 = tl.relu(branch7x7)
            with tf.variable_scope('branch7x7dbl'):
                branch7x7dbl = tl.conv2d(net, 192, (1, 1),name='branch7x7dbl_1')
                branch7x7dbl = tl.bn_new(branch7x7dbl,name='branch7x7dbl_1')
                branch7x7dbl = tl.relu(branch7x7dbl)
                branch7x7dbl = tl.conv2d(branch7x7dbl, 192, (7, 1),name='branch7x7dbl_2')
                branch7x7dbl = tl.bn_new(branch7x7dbl,name='branch7x7dbl_2')
                branch7x7dbl = tl.relu(branch7x7dbl)
                branch7x7dbl = tl.conv2d(branch7x7dbl, 192, (1, 7),name='branch7x7dbl_3')
                branch7x7dbl = tl.bn_new(branch7x7dbl,name='branch7x7dbl_3')
                branch7x7dbl = tl.relu(branch7x7dbl)
                branch7x7dbl = tl.conv2d(branch7x7dbl, 192, (7, 1),name='branch7x7dbl_4')
                branch7x7dbl = tl.bn_new(branch7x7dbl,name='branch7x7dbl_4')
                branch7x7dbl = tl.relu(branch7x7dbl)
                branch7x7dbl = tl.conv2d(branch7x7dbl, 192, (1, 7),name='branch7x7dbl_5')
                branch7x7dbl = tl.bn_new(branch7x7dbl,name='branch7x7dbl_5')
                branch7x7dbl = tl.relu(branch7x7dbl)
            with tf.variable_scope('branch_pool'):
                branch_pool = tl.avg_pool2d(net, (3, 3))
                branch_pool = tl.conv2d(branch_pool, 192, (1, 1))
                branch_pool = tl.bn_new(branch_pool)
                branch_pool = tl.relu(branch_pool)
                net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool],3)
        return net

    def Inceptionv3_module3(self, net,name):
        with tf.variable_scope(name) as scope:

            with tf.variable_scope('branch1x1'):
                branch1x1 = tl.conv2d(net, 320, (1, 1))
                branch1x1 = tl.bn_new(branch1x1)
                branch1x1 = tl.relu(branch1x1)
            with tf.variable_scope('branch3x3'):
                branch3x3 = tl.conv2d(net, 384, (1, 1),name='branch3x3_0')
                branch3x3 = tl.bn_new(branch3x3)
                branch3x3 = tl.relu(branch3x3)
                branch3x3 = tf.concat([tl.conv2d(branch3x3, 384, (1, 3),name='branch3x3_1'),
                                       tl.conv2d(branch3x3, 384, (3, 1),name='branch3x3_2')],3)
            with tf.variable_scope('branch3x3dbl'):
                branch3x3dbl = tl.conv2d(net, 448, (1, 1),name='branch3x3dbl_1')
                branch3x3dbl = tl.bn_new(branch3x3dbl,name='branch3x3dbl_1')
                branch3x3dbl = tl.relu(branch3x3dbl)
                branch3x3dbl = tl.conv2d(branch3x3dbl, 384, (3, 3),name='branch3x3dbl_2')
                branch3x3dbl = tl.bn_new(branch3x3dbl,name='branch3x3dbl_2')
                branch3x3dbl = tl.relu(branch3x3dbl)
                branch3x3dbl = tf.concat([tl.conv2d(branch3x3dbl, 384, (1, 3),name='branch3x3dbl_3'),
                                          tl.conv2d(branch3x3dbl, 384, (3, 1),name='branch3x3dbl_4')],3)
            with tf.variable_scope('branch_pool'):
                branch_pool = tl.avg_pool2d(net, (3, 3))
                branch_pool = tl.conv2d(branch_pool, 192, (1, 1))
                branch_pool = tl.bn_new(branch_pool)
                branch_pool = tl.relu(branch_pool)
                net = tf.concat( [branch1x1, branch3x3, branch3x3dbl, branch_pool],3)
        return net

    def inference_Inceptionv3(self,im):
        n=3
        # n_class=10
        n_class=101
        with tf.variable_scope('Inceptionv3') as scope:
            y = tl.conv2d(im, 32, (3, 3), (2, 2), name='conv0')
            y = tl.bn_new(y,name='conv1')
            y = tl.relu(y)
            y = tl.conv2d(y, 32, (3, 3), (1, 1), name='conv1')
            y = tl.bn_new(y,name='conv2')
            y = tl.relu(y)
            y = tl.conv2d(y, 64, (3, 3), (1, 1), padding='SAME', name='conv2')
            y = tl.bn_new(y,name='conv3')
            y = tl.relu(y)
            y = tl.max_pool2d(y, (3, 3), (2, 2), name='pool1')
            y = tl.bn_new(y,name='conv4')
            y = tl.relu(y)
            # 73 x 73 x 64
            y = tl.conv2d(y, 80, (3, 3), (1, 1), name='conv3')
            y = tl.bn_new(y,name='conv5')
            y = tl.relu(y)
            # 73 x 73 x 80.
            y = tl.conv2d(y, 192, (3, 3), (2, 2), name='conv4')
            y = tl.bn_new(y,name='conv6')
            y = tl.relu(y)
            # 71 x 71 x 192.
            y = tl.max_pool2d(y, (3, 3), (2, 2), name='pool2')
            # 35 x 35 x 192.
            for i in range(n - 1):
                y = self.Inceptionv3_module1(y, (5, 5),name='mixed_35x35x256a'+str(i))
            for i in range(n - 1):
                y = self.Inceptionv3_module2(y,name='mixed_17x17x768e'+str(i))
            for i in range(n - 1):
                y = self.Inceptionv3_module3(y,name='mixed_8x8x2048a'+str(i))

            shape = y.get_shape()
            y = tl.avg_pool2d(y, shape[1:3], padding='VALID', name='pool')
            # 1 x 1 x 2048
            y = tl.dropout(y, self.keep_prob)
            y = tl.flatten(y)
            # 2048
            logits = tl.fc(y, n_class, name='logits')
            return logits


    def MobileNet_separable_2d(self, net, out_channel, stride = (1, 1), name='separable'):
        width_multiplier=1
        net = tl.depthwise_conv2d(net, width_multiplier, (3, 3), stride, name = name + 'dep')
        net = tl.bn_new(net, name = name + 'dep')
        net = tl.relu(net)
        net = tl.conv2d(net, out_channel, (1,1), stride, name = name + 'conv')
        net = tl.bn_new(net, name = name + 'conv')
        net = tl.relu(net)
        return net

    def inference_MobileNet(self, im):
        n=6
        num_classes=10
        width_multiplier=1
        net = tl.conv2d(im, round(32 * width_multiplier), (3, 3), (2, 2), padding='SAME', name='conv_1')
        net = tl.bn_new(net, name='conv_1')
        net = tl.relu(net)
        net = self.MobileNet_separable_2d(net, 32, name ='sep1')
        net = self.MobileNet_separable_2d(net, 64, (2, 2), name='sep2')
        net = self.MobileNet_separable_2d(net, 128, name ='sep3')
        net = self.MobileNet_separable_2d(net, 128, (2, 2), name='sep4')
        net = self.MobileNet_separable_2d(net, 256, name ='sep5')
        net = self.MobileNet_separable_2d(net, 256, (2, 2), name='sep6')
        net = self.MobileNet_separable_2d(net, 512, name ='sep7')

        for i in range(n):
            net = self.MobileNet_separable_2d(net, 512, name ='sep' + str(i + 8))

        net = self.MobileNet_separable_2d(net, 512, (2, 2), name ='sep' + str(n + 9))
        net = self.MobileNet_separable_2d(net, 1024, name='sep' + str(n + 10))

        #shape = net.get_shape()
        #net = tl.avg_pool2d(net, shape[1:3], name='avg_pool')
        net = tf.reshape(net, [self.batch_size, -1])
        logits = tl.fc(net, num_classes, name='fc')
        return logits


    def compute_loss(self, labels, logits):
        labels = tf.cast(labels, tf.int32)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name = 'xentropy')
        xentropy = tf.reduce_mean(xentropy, name = 'xentropy_mean')
        return xentropy
		
    def compute_acc(self, labels, logits):
        labels = tf.cast(labels, tf.int64)
        acc = tf.equal(tf.argmax(logits, 1), labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc



