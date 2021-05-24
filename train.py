import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import unet_lstm
import utils
import json
import os
import math

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

batch_size = 16
epoch = 100

param = {
         'trial'        :   5  ,          ########## Modify before train ##########
         'model_name'   :   'unet_lstm',
         'batch'        :   batch_size,
         'epoch'        :   epoch,
        }
data_dir = 'data'

split_rate = 0.3
data_list = os.listdir(data_dir)
train_length = math.ceil(len(data_list)*(1-split_rate))
train_data_list = data_list[:train_length]
valid_data_list = data_list[train_length:]

train_generator = utils.DataGenerator(data_dir, train_data_list, batch_size)
valid_generator = utils.DataGenerator(data_dir, valid_data_list,  batch_size)
train_step_size = train_generator.n // train_generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size


save_file_name = "{}_t{}".format(param['model_name'], param['trial'])
history = tf.keras.callbacks.CSVLogger('./history/'+save_file_name+'_history.txt', separator="\t", append=True)
checkpoint = ModelCheckpoint('./weights/'+save_file_name+'.h5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

if __name__ == "__main__":
    model = unet_lstm.getModel()
    model.compile(optimizer='adam', loss=tf.keras.losses.log_cosh, metrics='acc')
    model.fit(train_generator,
              steps_per_epoch=train_step_size,
              epochs=epoch,
              verbose=1,
              validation_data=valid_generator,
              validation_steps=valid_step_size,
              callbacks=[history, checkpoint])
