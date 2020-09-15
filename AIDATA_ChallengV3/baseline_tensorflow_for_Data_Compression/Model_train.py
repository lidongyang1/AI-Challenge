import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import model_selection
from Model_define_tf import Encoder, Decoder, NMSE
#from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback,ReduceLROnPlateau

# parameters
feedback_bits = 128
img_height = 16
img_width = 32
img_channels = 2

# Model construction
# encoder model
Encoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="encoder_input")
Encoder_output = Encoder(Encoder_input, feedback_bits)
encoder = keras.Model(inputs=Encoder_input, outputs=Encoder_output, name='encoder')
#plot_model(encoder, to_file='encodermodel.png', show_shapes=True, show_layer_names=True) # plot my model

# decoder model
Decoder_input = keras.Input(shape=(feedback_bits,), name='decoder_input')
Decoder_output = Decoder(Decoder_input, feedback_bits)
decoder = keras.Model(inputs=Decoder_input, outputs=Decoder_output, name="decoder")
#plot_model(decoder, to_file='decodermodel.png', show_shapes=True, show_layer_names=True) # plot my model

# autoencoder model
autoencoder_input = keras.Input(shape=(img_height, img_width, img_channels), name="original_img")
encoder_out = encoder(autoencoder_input)
decoder_out = decoder(encoder_out)
autoencoder = keras.Model(inputs=autoencoder_input, outputs=decoder_out, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')
print(autoencoder.summary())

# Data loading
data_load_address = './data'
mat = h5py.File(data_load_address+'/Hdata.mat', 'r')
data = np.transpose(mat['H_train'])      # shape=(320000, 1024)
data = data.astype('float32')
data = np.reshape(data, [len(data), img_channels, img_height, img_width])
data = np.transpose(data, (0, 2, 3, 1))   # change to data_form: 'channel_last'
x_train, x_test = model_selection.train_test_split(data, test_size=0.05, random_state=1)

def evaluate(): # 评测函数
    y_test = autoencoder.predict(x_test, batch_size=512)
    return NMSE(x_test, y_test) # 爱算啥就算啥
# 定义Callback器，计算验证集的acc，并保存最优模型
class Evaluate(Callback):

    def on_train_begin(self, logs=None):
        self.losses = []
        self.best = 10.

    def on_epoch_end(self, epoch, logs=None):
        loss = evaluate()
        self.losses.append(logs.get('nmse:'+str(loss)))
        if loss <= self.best: # 保存最优模型权重
            self.best = loss
            encoder.save('./Modelsave/best_encoder_model.h5')
            decoder.save('./Modelsave/best_decoder_model.h5')
            np.savetxt('./Results/nmse',np.array([self.best]))
        print('\nnmse:'+str(self.best))
        return self.best
evaluator = Evaluate()
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9,
                              patience=5, min_lr=0.00001)
# model training
autoencoder.fit(x=x_train, y=x_train, batch_size=128, epochs=1000, validation_split=0.1,verbose=1,callbacks=[evaluator,reduce_lr])

# model save
# save encoder
modelsave1 = './Modelsave/encoder.h5'
encoder.save(modelsave1)
# save decoder
modelsave2 = './Modelsave/decoder.h5'
decoder.save(modelsave2)

# model test
y_test = autoencoder.predict(x_test, batch_size=512)
print('The NMSE is ' + np.str(NMSE(x_test, y_test)))







