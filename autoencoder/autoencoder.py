from deeparpes.autoencoder.noise_generator import *
from deeparpes.autoencoder.data_generator import *
import tensorflow as tf
import keras
from keras.metrics import mean_squared_error
from keras import layers

# x_train = np.concatenate([multigen(6000, 1), multigen(3000, 2, factors = [1, 0.5]), multigen(3000, 3, factors = [1, 0.5, 0.5])])
# x_test = np.concatenate([multigen(150, 1), multigen(150, 2, factors = [1, 0.5]), multigen(150, 3, factors = [1, 0.5, 0.5])])

# x_train = np.concatenate([gen(3000))])
# x_test = gen(100)

print('Generating data')
x_train = gen(6000)
x_test = gen(100)
print('Data generation complete')

np.random.shuffle(x_train)
np.random.shuffle(x_test)

print('Adding Noise')
x_train = reshape(x_train)
x_test = reshape(x_test)

x_train_noisy = noisy(x_train)
x_test_noisy = noisy(x_test)
print('Noise Complete')

import matplotlib.pyplot as plt

n = 10

# d = norm_ln_batch(data[12])
# d = norm_batch(data[14])

# sanity check
print("showing testing data")
try:
  plt.figure(figsize=(20, 6))
  for i in range(1, n + 1):
      ax = plt.subplot(3, n, i)
      plt.imshow(x_train[i].reshape(h, w))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      ax = plt.subplot(3, n, i+n)
      plt.imshow(x_train_noisy[i].reshape(h, w))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

      ax = plt.subplot(3, n, i+2*n)
      plt.imshow(x_test[i].reshape(h, w))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()
except:
  pass

x_train = reshape(x_train)
x_test = reshape(x_test)

x_train_noisy = noisy(x_train)
x_test_noisy = noisy(x_test)

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def joint_MAE_SSIMLoss(y_true, y_pred):
  return 0.8*mean_squared_error(y_true, y_pred) + 0.2*SSIMLoss(y_true, y_pred)

input_img = keras.Input(shape=(h, w, 1))

x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((4, 4), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# At this point the representation is (8, 8, 32)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((4, 4))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=joint_MAE_SSIMLoss)
autoencoder.summary()
