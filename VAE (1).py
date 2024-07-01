# %% [code] {"jupyter":{"outputs_hidden":false}}
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import mse
from tensorflow.keras.models import load_model, save_model

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, LeakyReLU, Conv2D, MaxPooling2D, Activation, Input, UpSampling1D, BatchNormalization, Lambda, Concatenate, Layer, Embedding, Conv2DTranspose
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import os
from matplotlib import pyplot as plt



weight = K.variable(0.)
    
class AnnealingCallback(Callback):
    def __init__(self, weight, klstart, kl_annealtime):
        
        self.weight = weight
        K.set_value(self.weight, 0)
        self.klstart = klstart
        self.kl_annealtime = kl_annealtime
        
    def on_epoch_end (self, epoch, logs={}):
        if epoch >= self.klstart :
            new_weight = min(K.get_value(self.weight) + (1./ self.kl_annealtime ), 1.)
            K.set_value(self.weight, new_weight)
        # print ("Current KL Weight is " + str(K.get_value(self.weight)))

        


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        shapes = tf.shape(z_mean)
        # print(f"Tesnor shapes - {shapes}")
        batch = shapes[0]
        dim = shapes[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
      
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, train_data):

        data = train_data[0][0];
        labels = train_data[0][1];

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([data, labels])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    mse(data, reconstruction), axis=(1, 2)
                )
            ) #mse is a better loss for this, than bce
           
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = kl_loss * weight
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def get_encoder(trial_shape, num_classes, latent_dim):

  kernel_size_1 = (1, (2 * trial_shape[1])//96)
  strides_1 = (1, np.max((trial_shape[1]//96, 4)))
  kernel_size_2 = (trial_shape[0], 1)
  
  encoder_inputs = Input(shape=trial_shape)
  in_label = Input(shape=(num_classes,))

  label_embedding = Embedding(num_classes, 10)(in_label)
  label_embedding = Flatten()(label_embedding)
  label_embedding = Dense(5)(label_embedding)
  n_nodes = np.prod(trial_shape) 

  label_embedding = Dense(n_nodes)(label_embedding)
  label_embedding = Reshape(target_shape=(trial_shape))(label_embedding)

  inputs = Concatenate()([encoder_inputs, label_embedding]) 
  hidden_l = Conv2D(16, kernel_size_1, strides=strides_1)(inputs)
  hidden_l = LeakyReLU(alpha=0.2)(hidden_l)
  hidden_l = Conv2D(32, kernel_size_2)(hidden_l)
  hidden_l = LeakyReLU(alpha=0.2)(hidden_l)

  # shape info for building decoder
  shape = K.int_shape(hidden_l)

  hidden_l = Flatten()(hidden_l)
  hidden_l = Dense(16)(hidden_l)

  z_mean = Dense(latent_dim, name="z_mean")(hidden_l)
  z_log_var = Dense(latent_dim, name="z_log_var")(hidden_l)
  z = Sampling()([z_mean, z_log_var])

  encoder = Model(inputs=[encoder_inputs, in_label], outputs=[z_mean, z_log_var, z], name="encoder")
  encoder.summary()
  return encoder, shape 



def get_decoder(trial_shape, latent_dim, intermediate_shape):

  kernel_size_1 = (trial_shape[0], 1)
  kernel_size_2 = (1, (2 * trial_shape[1])//96)
  strides_2 = (1, np.max((trial_shape[1]//96, 4)))
  
  # decoder - strides is very impt for convTr
  latent_inputs = Input(shape=(latent_dim,))
  decoder_hidden = Dense(intermediate_shape[1] * intermediate_shape[2] * intermediate_shape[3])(latent_inputs)
  decoder_hidden = LeakyReLU(alpha=0.2)(decoder_hidden)
  decoder_hidden = Reshape((intermediate_shape[1], intermediate_shape[2], intermediate_shape[3]))(decoder_hidden)
  decoder_hidden = Conv2DTranspose(64, kernel_size_1)(decoder_hidden)
  decoder_hidden = LeakyReLU(alpha=0.2)(decoder_hidden)
  decoder_hidden = Conv2DTranspose(32, kernel_size_2, strides=strides_2)(decoder_hidden)
  decoder_hidden = LeakyReLU(alpha=0.2)(decoder_hidden)
  decoder_outputs = Conv2DTranspose(1, 7, padding="same")(decoder_hidden)
  decoder_outputs = LeakyReLU(alpha=0.2)(decoder_outputs)
  decoder = Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
  decoder.summary()

  return decoder
  
def augment_data(data, labels, with_noise=False, n_augs = 1000):
  # augment train by adding replicas with noise - use std to be sure you're not corrupting the data with std of 1. e.g cho's was /1e-6.
  # check the range, mean... of data 
  
  data = np.squeeze(data)

  data_shape = data.shape
  new_data = data[:]
  new_y_labels = labels[:]

  while len(new_data) < n_augs:
    aug_data = data[:]
    new_data = np.concatenate((new_data, aug_data))
    new_y_labels = np.concatenate((new_y_labels, labels)) 

  new_data = np.expand_dims(new_data, -1)
  return new_data, new_y_labels


def train_vae(data, labels, save_path, epochs=500):
   
  data, labels = augment_data(data, labels)
  
  latent_dim = 10
  trial_shape = data[0].shape
  num_classes = len(np.unique(labels))
  labels_encoded, _ = encode_labels(labels, num_classes)

  encoder, intermediate_shape = get_encoder(trial_shape, num_classes, latent_dim)
  decoder = get_decoder(trial_shape, latent_dim, intermediate_shape)
  
  # callback
#   weight = K.variable(0.)
  klstart = 40  #80
  # number of epochs over which KL scaling is increased from 0 to 1
  kl_annealtime = 20

  anneal_cbk = AnnealingCallback(weight, klstart, kl_annealtime)

  # VAE
  vae = VAE(encoder, decoder)
  vae.compile(optimizer=Adam()) 
  history = vae.fit(x=[data, labels_encoded], epochs=epochs, batch_size=16, callbacks=[anneal_cbk])     #500

  encoder_save_path = os.path.join(save_path, 'encoder')
  decoder_save_path = os.path.join(save_path, 'decoder')
  if not os.path.exists(encoder_save_path):
    os.makedirs(encoder_save_path)
  if not os.path.exists(decoder_save_path):
    os.makedirs(decoder_save_path)
    
  save_model(vae.encoder, encoder_save_path)
  save_model(vae.decoder, decoder_save_path)

  
  # plot history
  reconstruction_loss = history.history['reconstruction_loss']
  kl_loss = history.history['kl_loss']
  total_loss = history.history['loss']
  epochs = list(range(len(reconstruction_loss)))

  plt.plot(epochs, reconstruction_loss, 'r--', epochs, kl_loss, 'g--', epochs, total_loss, 'b--')
  plt.legend(['Reconstruction Loss', 'KL Loss', 'Total' ])
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.savefig(f'{save_path}/losses.png')
  plt.close('all')


    


def plot_gend_data(data, gend_samples, n_to_show = 12):
  
  fig = plt.figure(figsize=(15, 3))
  fig.subplots_adjust(hspace=0.4, wspace=0.4)

  for i in range(n_to_show):
    img = data[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+1)
    sub.axis('off')  
    sub.imshow(img, extent=[0,100,0,1], aspect='auto')

  for i in range(n_to_show):
    img = gend_samples[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
    sub.axis('off')
    sub.imshow(img, extent=[0,100,0,1], aspect='auto')  



def generate_data_from_vae(subject, mapped_data_labels, n_gen, master_path, data_paths):
  
  encoder_save_path = os.path.join(master_path, data_paths['saved_models'], 'vaes', subject, 'encoder')
  decoder_save_path = os.path.join(master_path, data_paths['saved_models'], 'vaes', subject, 'decoder')

  encoder = load_model(encoder_save_path)
  decoder = load_model(decoder_save_path)
  
  labels = list(mapped_data_labels.keys())
  synth_data_labels = {}

  for label in labels:
    num_gen = round(n_gen/len(labels))
    data_for_label = mapped_data_labels[label]
    num_data = len(data_for_label)
    num_classes = len(np.unique(labels))
    data_gendfor_label = []

    while num_gen > 0:
      # randomly choose 'num_gen' of data & reconstruct from vae
      combination_choices = np.random.choice(num_data, size=np.min([num_data, num_gen]))
      choices_data = data_for_label[combination_choices]
      n_chosen = len(combination_choices)

      class_labels = np.asarray([1 if label == lab else 0 for lab in labels])
      class_labels = np.asarray([class_labels for i in range(n_chosen)])

      _, _, gend_latent = encoder.predict([choices_data, class_labels])
      gend_samples = decoder.predict(gend_latent)

      real_plot_data = choices_data
      synth_plot_data = gend_samples

      data_gendfor_label.append(gend_samples)
      num_gen -= n_chosen

    data_gendfor_label = np.concatenate(data_gendfor_label) 
    data_gendfor_label = np.squeeze(data_gendfor_label)
    synth_data_labels[label] = data_gendfor_label
  # plot_gend_data(real_plot_data, synth_plot_data, n_to_show = 12)

  return synth_data_labels
  
def encode_labels(y, num_classes):
    encoder = LabelEncoder()
    encoder.fit(y)
    y_enc = encoder.transform(y)
    
    return to_categorical(y_enc, num_classes), encoder