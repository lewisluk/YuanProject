#!/usr/bin/env python
# coding: utf-8

# In[13]:



from __future__ import print_function, division

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import RMSprop
from functools import partial

from tensorflow.python.keras import backend as K
# import tensorflow.python.keras.backend as K

import matplotlib.pyplot as plt

import sys
import math 
import numpy as np


# In[14]:



#Data preprocessing
train=[]
for i in range(1,631):
    fname=str(i)
    #Enter strawberry folder address
    with open("strawberries/negatives/text" + fname + ".txt", "r") as f:

          xy =f.read().split("\n")
          x=[]
          y=[]
          for i in range(0,len(xy)-1):
             x .append(xy[i].split("\t")[0])
             y .append(xy[i].split("\t")[1])
          train.append([x,y])

train =np.array(train).astype(np.float64)            
             
tra =np.arctan(train)*2/math.pi       
trains=[]
for  i in range(0,len(tra)):
   trains.append(np.array([tra[i]]))
trains=np.array(trains)
trains=trains.transpose(0,2,3,1)


# In[ ]:





# In[15]:




class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((90, 2, 235, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


# In[16]:


class WGANGP():
    def __init__(self):
        self.img_rows =2
        self.img_cols =235
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 470
 
        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00002)
 
        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_critic()
 
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------
 
        # Freeze generator's layers while training critic
        self.generator.trainable = False
 
        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
 
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)
 
        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
 
        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)
 
        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names
 
        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[2, 2, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------
 
        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True
 
        # Sampled noise for input to generator
        z_gen = Input(shape=(470,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
 
 
    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)
 
 
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
 
    def build_generator(self):
 
        model = Sequential()
 
        model.add(Dense(235 * 2 * 1, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((2, 235, 1)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels,2, strides=4))
        model.add(Activation("relu"))

 
        model.summary()
 
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
 
        return Model(noise, img)
 
    def build_critic(self):
 
        model = Sequential()
 
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
 
        model.summary()
 
        img = Input(shape=self.img_shape)
        validity = model(img)
 
        return Model(img, validity)
 
    def train(self, epochs, batch_size, sample_interval=50):
 
        # Load the dataset
        X_train= trains
 
       
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):
          
            for _ in range(self.n_critic):
 
                # ---------------------
                #  Train Discriminator
                # ---------------------
 
                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])
 
            # ---------------------
            #  Train Generator
            # ---------------------
 
            g_loss = self.generator_model.train_on_batch(noise, valid)
 
            # Plot the progress
          
 
            # If at save interval => save generated image samples
            if epoch % 200 == 0:
                self.sample_images(epoch)
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
 
    def sample_images(self, epoch):
        r, c =1,1
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
 
        
        

        generated_images=gen_imgs[0].transpose(2,0,1)
        plt.plot( generated_images[0][0],generated_images[0][1],"-")
        
        plt.pause(4)

        


# In[17]:


w =WGANGP()


# In[74]:



#train
w.train(50000,90)


# In[97]:



#predict
'''noise = np.random.normal(0, 1, (1, w.latent_dim))
gen_imgs = w.generator.predict(noise)
generated_images=gen_imgs[0].transpose(2,0,1)
plt.plot( generated_images[0][0],generated_images[0][1],"-")'''




# In[ ]:









# In[ ]:




