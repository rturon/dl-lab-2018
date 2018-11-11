import numpy as np
import tensorflow as tf

from cnn_mnist import mnist, train_and_validate

train_x, train_y, val_x, val_y, test_x, test_y = mnist('../exercise1/data')

train_x = train_x[:10000]
print(train_x.shape)
train_y = train_y[:10000]
val_x = val_x[:1000]
val_y = val_y[:1000]

_, model = train_and_validate(train_x, train_y, val_x, val_y,
                              num_epochs=10,
                              lr=0.01,
                              num_filters=10,
                              batch_size=64)
