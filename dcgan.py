# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from datetime import datetime

import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
 
# define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	# warstwa konwolucji, przeksztalcenie obrazu 28x28x1 do tensora 14x14x32
	model.add(Conv2D(32, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	# warstwa konwolucji, przeksztalcenie tensora 14x14x32 do 7x7x64 
	model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	# warstwa konwolucji, przeksztalcenie tensora 7x7x64 do 3x3x128
	model.add(Conv2D(128, (3,3), strides=(2, 2), padding='same'))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	# splaszczenie tensora
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# kompilacja modelu
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# podstawa obrazu 7x7
	n_nodes = 256 * 7 * 7
	# przeksztalcenie wejscia na tensor o wymiarach 7x7x256
	model.add(Dense(n_nodes, input_dim=latent_dim))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 256)))
	# uzycie transponowanej konwolucji do przeksztalcenia 7x7x256 do 14x14x128
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	# uzycie transponowanej konwolucji do przeksztalcenia 7x7x256 do 14x14x64
	model.add(Conv2DTranspose(64, (4,4), strides=(2,2), padding='same'))
	# zastosowanie funkcji aktywacji
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(1, (7,7), activation='sigmoid', padding='same'))
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# ustawienie wag dyskryminatora na nietrenowalne
	d_model.trainable = False
	# polaczenie generatora i dyskryminatora
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	# budowa i kompilacja modelu 
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# load and prepare mnist training images
def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y
 
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# iteracja epok
	for i in range(n_epochs):
		# iteracja po porcjach danych ze zbiory treningowego
		for j in range(bat_per_epo):
			# pobranie losowych probek 'prawdziwych'
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generowanie probek 'falszywych'
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# utworzenie zbioru treningowego dla dyskryminatora
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# aktualizacja wag dyskryminatora
			d_loss, _ = d_model.train_on_batch(X, y)
			# przygotowanie punktow w przestrzeni jako wejscie dla generatora
			X_gan = generate_latent_points(latent_dim, n_batch)
			# utworzenie etykiet dla falszywych probek
			y_gan = ones((n_batch, 1))
			# ulepszenie generatora na podstawie bledu dyskryminatora
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# podsumowanie straty w danej porcji danych
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# sprawdzenie wydajnosci modelu
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)

start_time = datetime.now().replace(microsecond=0) 
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
end_time = datetime.now().replace(microsecond=0)
total_time = (end_time - start_time)
print("TIME TAKEN: {}".format(total_time))
