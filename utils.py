from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import numpy as np
import random
import math
import os


class DataGenerator(Sequence):
    def __init__(self, data_dir, data_list, batch_size, mode='train', shuffle=True):
        self.data_dir = data_dir
        self.data_list = data_list
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.n = len(self.data_list)

        if self.shuffle and (self.mode == 'train'):
            random.shuffle(self.data_list)

    def __len__(self):
        return math.ceil(len(self.data_list)/self.batch_size)

    def __getitem__(self, index):
        data_batch = self.data_list[index * self.batch_size : (index + 1) * self.batch_size]
        xs, ys = self.__data_generation(data_batch)

        if self.mode == 'train':
            return xs, ys
        else:
            return xs

    def __data_generation(self, batch_list):
        x, y = [[] for i in range(len(batch_list))], [[] for i in range(len(batch_list))]
        for idx, filename in enumerate(batch_list):
            dataxy = np.array(np.load(os.path.join(self.data_dir, filename)))
            datax = dataxy[:,:,:4]
            x[idx] = datax / 255.
            if self.mode == 'train':
                datay = dataxy[:,:,4]
                y[idx] = datay / 255.
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        return x, y

class Prediction:
    def __init__(self, data_gene, batch_size, model, output_dir=None, save_bool=False):
        self.data_gene = data_gene
        self.batch_size = batch_size
        self.model = model
        self.pred = []
        self.output_dir = output_dir
        self.save_bool = save_bool
        self.__prediction(self.model)

    def get_predict(self):
        return self.pred

    def __prediction(self, model):
        pred = model.predict(self.data_gene, batch_size=self.batch_size, verbose=1)
        self.pred = pred
        if self.save_bool:
            for idx, img in tqdm(enumerate(pred)):
                img *= 255
                img = (np.where(img<0, 0 ,img).astype(dtype=np.uint8)).reshape((120,120))
                pred_img = Image.fromarray(img)
                fname, extension = self.data_gene.data_list[idx].split('.')
                pred_img.save(os.path.join(self.output_dir, fname + ".png"))

        print("Prediction End")

