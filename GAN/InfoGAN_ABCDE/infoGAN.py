# The code is shared on SDSC Github
# Description:
# generate English letter A,B,C,D and E using Information Maximizing GAN

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # used to generate grid images
import os
import random

max_epoch = 1000
batch_size = 128
Z_dim = 16 # length of latent vector of noise

# define a function to initialize random numbers for all weights
def infoGAN_init(size):
    return tf.random_normal(shape=size, stddev=0.1)

D_W1 = tf.Variable(infoGAN_init([784, 128])) # size -> [784 pixels, 128 hidden units]
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(infoGAN_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2] # parameters for discriminator

G_W1 = tf.Variable(infoGAN_init([21, 256]))
G_b1 = tf.Variable(tf.zeros(shape=[256]))

G_W2 = tf.Variable(infoGAN_init([256, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2] # parameters for generator

Q_W2 = tf.Variable(infoGAN_init([128, 5]))
Q_b2 = tf.Variable(tf.zeros(shape=[5]))

theta_Q = [Q_W2, Q_b2]

# randomly initialize latent vectors
def sample_Z(m, n): # （m=128, n=16）; 128 is batch size
    return np.random.uniform(-1., 1., size=[m, n])

# generate latent code; one-hot categorical code in this example
def sample_c(m):
    return np.random.multinomial(1, 5*[0.1], size=m) # for 5 categories
# for example, 'np.random.multinomial(1, 5*[0.1], size=3)' will return
# [[0,0,0,0,1],
#  [0,0,0,0,1],
#  [0,0,1,0,0]]
# *noted that the positions of '1' are random

def generator(z, c):
    # z -> [128, 16]
    # c -> [128, 5]
    inputs = tf.concat(axis=1, values=[z, c])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

def Q(x):
    # share the weights and biases of discriminator (except for output layer)
    Q_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    Q_prob = tf.nn.softmax(tf.matmul(Q_h1, Q_W2) + Q_b2)
    return Q_prob

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

X = tf.placeholder(tf.float32, shape=[None, 784])
Z = tf.placeholder(tf.float32, shape=[None, 16])
c = tf.placeholder(tf.float32, shape=[None, 5])

G_sample = generator(Z, c)
# z -> [batch_size(128), 16]
# c -> [batch_size(128), 5]

D_real = discriminator(X)
D_fake = discriminator(G_sample)
Q_c_given_x = Q(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real + 1e-8) + tf.log(1 - D_fake + 1e-8))
G_loss = -tf.reduce_mean(tf.log(D_fake + 1e-8))

Q_loss = tf.reduce_mean(-tf.reduce_sum(tf.log(Q_c_given_x + 1e-8) * c, 1))
# compare with their randomly generated labels, 'c'

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
Q_solver = tf.train.AdamOptimizer().minimize(Q_loss, var_list=theta_G + theta_Q)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('infoGAN_result/'):
    os.makedirs('infoGAN_result/')

i = 0

data = np.float32(np.loadtxt('imgvectors.txt'))

for e in range(max_epoch):
    for it in range(data.shape[0]//batch_size):
        X_mb = data[it*batch_size:(it+1)*batch_size,:]

        Z_noise = sample_Z(batch_size, Z_dim)
        c_noise = sample_c(batch_size)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_noise, c: c_noise})

        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_noise, c: c_noise})

        sess.run([Q_solver], feed_dict={Z: Z_noise, c: c_noise})

    if e % 20 == 0:
        print('Epcoh: {}'.format(e))
        Z_noise = sample_Z(25, Z_dim) # Z_dim = 16
        c_noise = np.zeros([25, 5]) # for 5 categories

        # option 1: show different categories
        c_noise[0:5,0] = 1
        c_noise[5:10,1] = 1
        c_noise[10:15,2] = 1
        c_noise[15:10,3] = 1
        c_noise[20:25,4] = 1

#        # option 2: show 1 category only
#        idx = np.random.randint(0,5)
#        c_noise[:,idx] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_noise, c: c_noise}) # generate results to show
        fig = plot(samples)
        plt.savefig('infoGAN_result/{}.png'.format(str(e).zfill(3)), bbox_inches='tight')
        plt.close(fig)

sess.close()
