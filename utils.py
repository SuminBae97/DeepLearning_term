from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import random
import math
import os


class DataGenerator(Sequence):
    def __init__(self, data_dir, data_list, batch_size, shuffle=True):
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.data_list)

        if self.shuffle:
            random.shuffle(self.data_list)

    def __len__(self):
        return math.ceil(len(self.data_list)/self.batch_size)

    def __getitem__(self, index):
        data_batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        xs, ys = self.__data_generation(data_batch)

        return xs, ys

    def __data_generation(self, batch_list):
        x, y = [[] for i in range(len(batch_list))], [[] for i in range(len(batch_list))]
        for idx, filename in enumerate(batch_list):
            dataxy = np.array(np.load(os.path.join(self.data_dir, filename)))
            datax = dataxy[:,:,:4]
            datay = dataxy[:,:,4]
            x[idx] = datax / 255.
            y[idx] = datay / 255.
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        return x, y





