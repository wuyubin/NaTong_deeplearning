import os
import numpy as np

class data_layer:
    is_dummy = False

    def __init__(self, FLAGS, type):
        self.batch_size = FLAGS.batch_size
        if type is 'train' or 'valid':
            self.data_path = os.path.join(FLAGS.data_path, type + '_data.npz')
            # self.label_path = os.path.join(FLAGS.data_path, type + '_label.npy')
        else:
            ValueError('type must be train or valid')
        self.start_idx = 0

        self._load_data()
        self._epoch = -1
        self._iteration = 0
        self._iter_cur_epoch = 0

    def _load_data(self):
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

    def next_batch(self):
        if self.start_idx == 0:
            self._epoch += 1
            self._iter_cur_epoch = 0
        s = self.start_idx
        e = min(self.data.shape[0], s + self.batch_size)
        data_batch = self.data[s:e]
        label_batch = self.label[s:e]

        self._iteration += 1
        self._iter_cur_epoch += 1
        if e == self.data.shape[0]:
            self.start_idx = 0
        else:
            self.start_idx = e
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


