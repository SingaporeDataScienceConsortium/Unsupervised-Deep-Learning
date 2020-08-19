# The code is shared on SDSC Github
# Description:
# generate English letter A,B,C,D and E using conditional GAN

import tensorflow as tf
import numpy as np
import os
import shutil
import imageio

img_height = 28  # image height
img_width = 28   # image width
img_size = img_height * img_width   # total number of pixels in each image

max_epoch = 500

h1_size = 150     # 1st hidden layer
h2_size = 300     # 2nd hidden layer
z_size = 100      # length of the latent vector
y_size=5         # number of categories/conditions
batch_size = 128

# generator
def build_generator(z_prior,y):
    # size of z_prior -> [128, 100]
    # size of y -> [128, 5]
    inputs = tf.concat(axis=1, values=[z_prior, y]) # size -> [128, 105]

	# 1st layer
    w1 = tf.Variable(tf.truncated_normal([z_size+y_size, h1_size], stddev=0.1), dtype=tf.float32) # size -> [105, 150]
    b1 = tf.Variable(tf.zeros([h1_size]), dtype=tf.float32) # size -> 150
    h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)

    # 2nd layer
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), dtype=tf.float32) # size -> [150, 300]
    b2 = tf.Variable(tf.zeros([h2_size]), dtype=tf.float32) # size -> 300
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    # output layer
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), dtype=tf.float32) # size -> [300, 784]
    b3 = tf.Variable(tf.zeros([img_size]), dtype=tf.float32) # size -> 784
    h3 = tf.matmul(h2, w3) + b3
    x_generate = tf.nn.tanh(h3)
    g_params = [w1, b1, w2, b2, w3, b3] # integrate trainable parameters for generator
    return x_generate, g_params
    # size of x_generate -> [128, 784]


# discriminator
def build_discriminator(x_data, x_generated,y, keep_prob):
    data_and_y = tf.concat(axis=1, values=[x_data, y])  # size = [batch_size,784+5]
    gen_and_y = tf.concat(axis=1, values=[x_generated, y])  # size = [batch_size,784+5]

    x_in = tf.concat([data_and_y, gen_and_y], 0)
    # the first half is real (top) data and the second (bottom) half is generated data

    # 1st layer
    w1 = tf.Variable(tf.truncated_normal([img_size+y_size, h2_size], stddev=0.1), dtype=tf.float32) # size -> [784+5, 300]
    b1 = tf.Variable(tf.zeros([h2_size]), dtype=tf.float32) # size -> 300
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)

    # 2nd layer
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), dtype=tf.float32) # size -> [300, 150]
    b2 = tf.Variable(tf.zeros([h1_size]), dtype=tf.float32) # size -> 150
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

    # output layer
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), dtype=tf.float32) # size -> [150, 1]
    b3 = tf.Variable(tf.zeros([1]), dtype=tf.float32) # size -> 1
    h3 = tf.matmul(h2, w3) + b3

    # tf.slice is used to separate the first (top, for real data) half and the second (bottom, for generated data) half
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))

    d_params = [w1, b1, w2, b2, w3, b3]  # integrate trainable parameters for discriminator
    return y_data, y_generated, d_params

def train():
    x_data = tf.placeholder(tf.float32, [batch_size, img_size]) # [128, 784]
    y = tf.placeholder(tf.float32, shape=[batch_size, y_size]) # [128, 5]
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size]) # [128, 100]
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    x_generated, g_params = build_generator(z_prior,y)
    # z_prior -> [128, 100]
    # y -> [128, 5]
    # x_generated -> [128, 784]

    y_data, y_generated, d_params = build_discriminator(x_data, x_generated, y , keep_prob)
    # x_data -> [128, 784]
    # x_generated -> [128, 5]
    # y_data -> [128, 1]
    # y_generated -> [128, 1]

    d_loss = - (tf.log(y_data) + tf.log(1 - y_generated))
    g_loss = - tf.log(y_generated)

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    # create a folder to store results
    output_path = "CGAN_result"  # directory to save result
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    x_values = np.float32(np.loadtxt('imgvectors.txt'))
    y_labels = np.float32(np.loadtxt('labels.txt'))
    total_batch = int(x_values.shape[0]/batch_size) # number of batches

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Training cycle
    for i in range(max_epoch):
        for j in range(total_batch):
            x_value = x_values[j*batch_size:(j+1)*batch_size,:]
            y_label = y_labels[j*batch_size:(j+1)*batch_size,:]

            x_value=np.array(x_value)
            x_value = 2 * x_value.astype(np.float32) - 1 # normalize to -1 to 1
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32) # size -> [128, 100]

            sess.run(d_trainer,feed_dict={x_data: x_value,y:y_label,
                                          z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
            sess.run(g_trainer,feed_dict={x_data: x_value,y:y_label ,
                                          z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})

        if i%5==0:
            y_sample = np.zeros(shape=[batch_size, y_size]) # size -> [128, 5]
            y_sample[:5, 0] = 1    # label for category 1
            y_sample[5:10, 1] = 1  # label for category 2
            y_sample[10:15, 2] = 1 # label for category 3
            y_sample[15:20, 3] = 1 # label for category 4
            y_sample[20:, 4] = 1   # label for category 5

            z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            # for every epoch, 'z_random_sample_val' will be different

            x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val,y:y_sample})
            # size of x_gen_val -> [128, 784]

            show_result(x_gen_val, os.path.join(output_path, "sample%s.jpg" % i))
            print ("epoch:%s." % (i))
        sess.run(tf.assign(global_step, i + 1))
    sess.close()

# define a function to visualize the result
def show_result(batch_res, fname, grid_size=(5, 5), grid_pad=5):
    # row 1 -> category 1
    # row 2 -> category 2
    # row 3 -> category 3
    # row 4 -> category 4
    # row 5 -> category 5

    # normalize the values to 0-1
    batch_res = 0.5*batch_res.reshape((batch_res.shape[0], img_height, img_width))+0.5

    # create grid and show generated images
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]: # only show the first 5*5=25 images with pre-defined labels
            break
        img = (res) * 255 # convert image values to 0-255 for display
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imageio.imwrite(fname, img_grid)

if __name__ == '__main__':
    train()
