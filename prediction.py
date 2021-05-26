import tensorflow as tf
import unet_lstm
import numpy as np
import utils
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = 16
output_dir = 'output'
test_data_dir = 'data/test'
test_data_list = os.listdir(test_data_dir)
test_generator = utils.DataGenerator(test_data_dir, test_data_list, batch_size, mode='test')

model = unet_lstm.getModel()
weight = os.path.join('weights', 'unet_lstm_t2.h5')
model.load_weights(weight)

if __name__ == "__main__":
    pred = utils.Prediction(test_generator, batch_size, model, output_dir, True)
