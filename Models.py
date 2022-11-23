import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class ThreeLayersModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_layer    = Dense(4, input_shape=input_dim, activation='relu')
        self.hidden_1       = Dense(100, activation='relu')
        self.hidden_2       = Dense(100, activation='relu')
        self.hidden_3       = Dense(50, activation='relu')
        self.output_layer   = Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        return self.output_layer(x)

class FiveLayersModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_layer    = Dense(16, input_shape=input_dim, activation='relu')
        self.hidden_1       = Dense(64, activation='relu')
        self.hidden_2       = Dense(16, activation='relu')
        self.hidden_3       = Dense(32, activation='relu')
        self.hidden_4       = Dense(16, activation='relu')
        self.hidden_5       = Dense(16, activation='relu')
        self.output_layer   = Dense(output_dim, activation='linear')

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.hidden_4(x)
        x = self.hidden_5(x)
        return self.output_layer(x)
    
if __name__ == "__main__":
    model   = ThreeLayersModel((4, ),2)
    input = np.array([[1, 0, 0, 0]])
    print(input.shape)
    x = model(input)
    print(x)