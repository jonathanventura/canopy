from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, Lambda, MaxPooling2D, Cropping2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class PatchClassifier:
    def __init__(self,num_classes,init_num_filt=32,max_num_filt=128,num_layers=7,l2reg=0.001):
        reg = regularizers.l2(l2reg)

        self.convs = []
        num_filt = init_num_filt
        for i in range(num_layers):
          self.convs.append(Conv2D(num_filt,3,padding='valid',activation='relu',kernel_regularizer=reg))
          if num_filt < max_num_filt:
            num_filt *= 2
        self.convs.append(Conv2D(num_classes,1,activation='softmax',kernel_regularizer=reg))

    def get_convolutional_model(self,input_shape):
        inputs = Input(input_shape)
        x = inputs

        for conv in self.convs:
          x = conv(x)
        outputs = x

        return Model(inputs=inputs,outputs=outputs)

    def get_patch_model(self,input_shape):
        conv_model = self.get_convolutional_model(input_shape)

        inputs = Input(input_shape)
        x = inputs

        for conv in self.convs:
          x = conv(x)
        outputs = Flatten()(x)

        return Model(inputs=inputs,outputs=outputs)

