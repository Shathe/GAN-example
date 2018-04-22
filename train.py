from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np 
import cv2

mnist = input_data.read_data_sets('MNIST_data')
tf.reset_default_graph()
batch_size = 64
n_noise = 64

# input image (real)
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
# Noise input. The distribution to convert
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))
# Binary cross entroopy. Just for 2 different classes. Activation for this loss: sigmoid. (For more than 2 classes would be: Softmax)
def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# Define the discriminator. Classifyies between real/fake images
def discriminator(img_in, reuse=None, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x




# Define the generator. The generator transformes the input distribution into another one
def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = 0.99
    with tf.variable_scope("generator", reuse=None):
        x = z
        d1 = 4
        d2 = 1
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[7, 7])
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob, training=is_training)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=1, strides=1, padding='same', activation=tf.nn.sigmoid) #activation of 2 outputs, ifnot, softmax
        return x

# generator
g = generator(noise, keep_prob, is_training)
# Define the output of the discriminator when a X_in (real data) is given
d_real = discriminator(X_in) 

# Define the output of the discriminator when a output of the generator (fake data) is given
generated_images = g
d_fake = discriminator(generated_images, reuse=True) # Reuse the weights (when already define a tensorflow graph, in order to use the same, you need to define a reuse scope)

# The discriminator and generator and trained separately
vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")] # Get the variables/weights of the generator (in order to give them to the optimizer)
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]# Get the variables/weights of the discriminator (in order to give them to the optimizer)


# this can be ommited, this is a regularization which will be applied to the optimizer
d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

# Losses

# Discriminator loss. The discriminator have to distinguish between fake and real data (data distributions)
loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real) # loss of real data (the discriminator has to say always its real data (the input is always real data))
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake) # loss of fake data 
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake)) # The generator has to trick the discriminator (the discriminator have to say that the d_fake distribution is in fact, real )
loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake)) # mean of the 2 discriminator losses

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_d + d_reg, var_list=vars_d) # pass the cvariables to optimize
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.00015).minimize(loss_g + g_reg, var_list=vars_g) # pass the cvariables to optimize
    
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train the GAN
for i in range(600000):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    # input distribution noise
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)   
    # Real data
    batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]  
    
    # Check losses of the discrminator and the generator
    d_real_ls, d_fake_ls, g_ls, d_ls, imgs = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d, generated_images], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
    d_real_ls = np.mean(d_real_ls) 
    d_fake_ls = np.mean(d_fake_ls)
    # g_ls generator loss
    # d_ls discriminator loss 
    
    # Train them without having one more intelligent than the other
    if g_ls * 1.5 < d_ls:
        train_g = False
        pass
    if d_ls * 2 < g_ls:
        train_d = False
        pass

    if i % 5000 == 0:
    	cv2.imwrite(str(i)+'.png', imgs[0,:,:,:]*255)

    if i % 500 == 0:

	    print('Discriminator loss: ' + str(d_ls))
	    print('Generator loss: ' + str(g_ls))
    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        
        
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})