#importing necessary libraries
from tensorflow.keras.models import Model,Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,Input, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121

def Custom():
    densenet = DenseNet121(weights='imagenet', include_top=False)
    print(densenet.summary())

    input = Input(shape=(128, 128, 3))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(4,activation = 'softmax', name='root')(x)
 
    # model
    model = Model(input,output)
    
    return model