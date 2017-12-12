import os
import random
import numpy as np
class data_reader:
    def __init__(self, path):
        self.path = path

    def load_labelmap(self, filename):
        with open(filename) as f:
            lines = f.read().splitlines()
            fd2label = [None] * len(lines)
            label2fd = [None] * len(lines)
            for lb, line in enumerate(lines):
               s = line.split(' ')
               fd2label[lb] = float(s[1])
               label2fd[lb] = s[0]
        return np.array(label2fd), np.array(fd2label)

    def load_label_mat(self):
        label2fd, fd2label = self.load_labelmap(os.path.join(self.path))
        return label2fd, fd2label
