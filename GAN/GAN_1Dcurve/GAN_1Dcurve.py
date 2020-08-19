# The code is shared on SDSC Github
# Description:
# generate 1D plots

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph() # clear old information to restart training
plt.close("all")

# Hyper Parameters
batch_size = 64         # number of datasets
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_init_rand_num = 5             # number of initial random numbers for generator

raw = np.loadtxt('signals.txt') + 1     # load signals and add an offset
N_signals = raw.shape[0]                # number of signals in each dataset (72)
seq = np.arange(N_signals)              # get coordinates of time points (from 0 to 71)
seq_norm = (seq-np.mean(seq))/np.max(abs(seq-np.mean(seq))) # normalize the time points to -1 to 1
raw_matrix = np.vstack([raw for _ in range(batch_size)])    # create 64 (batch_size) copies of the basic signal
time_pts = np.vstack([seq_norm for _ in range(batch_size)]) # create 64 (batch_size) sets of time points

# generate real signals
def real_signals():
    noise = np.random.randint(1000, size=(batch_size, N_signals))/2500 # noise
    batch_signals = raw_matrix + noise # add noise to the basic signals
    return batch_signals

with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, N_init_rand_num])
    # number of rows:    64 (batch_size)
    # number of columns: 5  (N_init_rand_num)

    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    # number of rows:    64 (batch_size)
    # number of columns: 128  (number of neurons in the hidden layer)

    G_out = tf.layers.dense(G_l1, N_signals)
    # number of rows:    64 (batch_size)
    # number of columns: 72  (N_signals)

with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None, N_signals], name='real_in')
    # number of rows:    64 (batch_size)
    # number of columns: 72 (N_signals)

    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
    # number of rows:    64  (batch_size)
    # number of columns: 128 (number of neurons in the hidden layer)

    prob_real0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')
    # number of rows:    64 (batch_size)
    # number of columns: 1  (probability for real data)

    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)
    # number of rows:    64  (batch_size)
    # number of columns: 128 (number of neurons in the hidden layer)

    prob_real1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)
    # number of rows:    64 (batch_size)
    # number of columns: 1  (probability for fake data)

# calculate the loss for discriminator
# to minimize D_loss, prob_real0 should be close to 1 and prob_real1 should be close to 0
# however, our final goal is to make D_loss close to 0.5
D_loss = -tf.reduce_mean(tf.log(prob_real0) + tf.log(1-prob_real1))

# calculate the loss for generator
# to minimize G_loss, prob_real1 should be close to 1
G_loss = tf.reduce_mean(tf.log(1-prob_real1))

# where does 'adversarial' come from?
# as you can see that D_loss and G_loss have contrary aims. this is a game, or say adversarial.

# the networks for both discriminator and generator are simultaneously trained
train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

# initialize a session and all tensorflow variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())


N_iter = 5000 # number of iterations
for step in range(N_iter):

    # in each iteration, the real signals are newly generated
    signals = real_signals()

    # generate initial random numbers for generator
    G_init_rand_num = np.random.randn(batch_size, N_init_rand_num)
    # number of rows:    64 (batch_size)
    # number of columns: 5  (N_init_rand_num)

    # train and get results
    G_signals, prob_art0, Dl, Gl = sess.run([G_out, prob_real0, D_loss,  G_loss, train_D, train_G],
                                    {G_in: G_init_rand_num, real_art: signals})[:4]

    if step % 50 == 0:
        plt.cla()
        Ndisp = 10 # number of datasets to display
        rand_disp = np.random.randint(0,64,Ndisp) # randomly choose some datasets to display

        # display some real datasets and their mean
        [plt.plot(time_pts[0], signals[rand_idx], c = 'r', lw= 1) for rand_idx in rand_disp]
        plt.plot(time_pts[0], np.mean(signals,axis=0), c='r', lw=10, alpha=0.5, label='Real')

        # display some fake datasets and their mean
#        [plt.plot(time_pts[0], G_signals[rand_idx], c = 'g', lw= 1) for rand_idx in rand_disp] # optional: can uncomment this line to display
        plt.plot(time_pts[0], np.mean(G_signals,axis=0), c='g', lw=10, alpha=0.5, label='Generated')

        # show probability on the plot
        plt.text(-1.05, 4.4, 'Probability of real data=%.2f' % prob_art0.mean(), fontdict={'size': 15})
        print('Training %d/%d. Real Prob = %.4f. Discri Loss = %.4f. Gen Loss = %.4f.' % (step,N_iter,prob_art0.mean(),Dl,Gl))
        plt.ylim((-2, 5)); plt.legend(loc='upper right', fontsize=12); plt.pause(0.01)

