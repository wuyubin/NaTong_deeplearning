import os
import cv2
import numpy as np
from data_reader import data_reader
class data_layer:
    is_dummy = False
    def __init__(self, FLAGS, type):
        self.batch_size = FLAGS.batch_size
        self.data_type = FLAGS.data_type
        self.net_type = FLAGS.net_type
        self.type = type
        if self.data_type is 'cifar10':
           if type is 'train' or 'valid':
                self.data_path = os.path.join(FLAGS.data_path, type + '_data.npz')
                # self.label_path = os.path.join(FLAGS.data_path, type + '_label.npy')
           else:
                ValueError('type must be train or valid')
        elif self.data_type is 'ILSVRC12':
            if type is 'train' or 'valid':
                self.data_path = os.path.join(FLAGS.data_path, type + '.txt')
            else:
                ValueError('type must be train or valid')
        self._load_data(self.data_type)
        self.start_idx = 0
        self._epoch = -1
        self._iteration = 0
        self._iter_cur_epoch = 0

    def _load_data(self, data_type):
        # if data_type is 'cifar10':
        if data_type is 'natong':
            if self.is_dummy:
                N, H, W, C = 1000, 32, 32, 3
                self.data = np.zeros([N, H, W, C], dtype=np.uint8)
                self.label = np.zeros(N, dtype=np.int32)
            else:
                # self.data = np.load(self.data_path)
                # self.label = np.load(self.label_path)
                original_data = np.load(self.data_path)
                self.data = original_data["data"]
                self.label = original_data["label"]
            assert(self.data.shape[0] >= self.batch_size), 'batch_size too large!'
        elif data_type is 'ILSVRC12':
            if self.is_dummy:
                H, W, C = 277, 277, 3
                self.data = np.zeros([H, W, C], dtype=np.uint8)
                self.label = np.zeros(1, dtype=np.int32)
            else:
                data_read = data_reader(self.data_path)
                self.data, self.label = data_read.load_label_mat()
            assert (self.data.shape[0] >= self.batch_size), 'batch_size too large!'
    def next_batch(self):
        if self.start_idx == 0:
            self._epoch += 1
            self._iter_cur_epoch = 0
        s = self.start_idx
        #print(self.data.shape[0])
        e = min(self.data.shape[0], s + self.batch_size)
        data_batch_small = self.data[s:e]

        if self.net_type is 'AlexNet':
            data_batch = np.zeros((data_batch_small.shape[0], 277, 277, 3))
        elif self.net_type is 'VGG19' or 'VGG16' or 'ResNet':
            data_batch = np.zeros((data_batch_small.shape[0], 224, 224, 3))

        if self.data_type is 'cifar10':

            for data_i,i_data in enumerate(data_batch_small):
                if self.net_type is 'AlexNet':
                    #print(i_data.shape)
                    data_batch[data_i, :, :, :] = cv2.resize(i_data, (277, 277), interpolation=cv2.INTER_CUBIC)

                elif self.net_type is 'VGG19' or 'VGG16' or 'ResNet':
                    data_batch[data_i, :, :, :] = cv2.resize(i_data, (224, 224), interpolation=cv2.INTER_CUBIC)

        elif self.data_type is 'ILSVRC12':
            image_batch = self.data[s:e]

            if self.type is 'train':
                for data_i, data_str in enumerate(image_batch):
                    image_i = cv2.imread('/home/tgy/ImageNet/ILSVRC2012_img_train/train/' + data_str, cv2.IMREAD_COLOR)
                    data_batch[data_i, :, :, :] = np.array(cv2.resize(image_i, (277, 277), interpolation=cv2.INTER_CUBIC))
            else:
                for data_i, data_str in enumerate(image_batch):
                    image_i = cv2.imread('/home/tgy/ImageNet/ILSVRC2012_img_val/' + data_str, cv2.IMREAD_COLOR)
                    data_batch[data_i, :, :, :] = np.array(cv2.resize(image_i, (277, 277), interpolation=cv2.INTER_CUBIC))
        elif self.data_type is 'natong':
            data_batch = self.data[s:e]
        #data_batch = self.data[s:e]
        label_batch = self.label[s:e]
        if e == self.data.shape[0]:
            self.start_idx = 0
        else:
            self.start_idx = e
        self._iteration += 1
        self._iter_cur_epoch += 1
        #print self.start_idx
        return data_batch, label_batch

    def reset(self):
        self._iteration = 0
        self._epoch = 0

    def random_shuffle(self):
        N = self.data.shape[0]
        idx = np.random.permutation(N)
        self.data = self.data[idx]

    def shape(self):
        return self.data.shape

    @property
    def epoch(self):
        return self._epoch

    @property
    def iteration(self):
        return self._iteration

    @property
    def iter_cur_epoch(self):
        return self._iter_cur_epoch


