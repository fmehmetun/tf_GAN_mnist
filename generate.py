import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import cv2, math

generate_count = 64
example_count = 64
latent_dim = 100
weight_path = "model/gan_mnist"

# Generator.
def generator(inp):
	with tf.variable_scope("gen"):
		x1 = tf.layers.dense(inp, 7*7*64)
		x1 = tf.reshape(x1, (-1, 7, 7, 64))
		x1 = tf.layers.batch_normalization(x1, training=True)
		x1 = tf.nn.relu(x1)

		x2 = tf.layers.conv2d_transpose(x1, 32, kernel_size=5, strides=2, padding="same")
		x2 = tf.layers.batch_normalization(x2, training=True)
		x2 = tf.nn.relu(x2)

		x3 = tf.layers.conv2d_transpose(x2, 16, kernel_size=5, strides=2, padding="same")
		x3 = tf.layers.batch_normalization(x3, training=True)
		x3 = tf.nn.relu(x3)

		output_pred = tf.layers.conv2d_transpose(x3, 1, kernel_size=5, strides=1, padding="same")
		output_pred = tf.nn.sigmoid(output_pred)

		return tf.squeeze(output_pred, axis=-1)

gen_input = tf.placeholder(tf.float32, shape=(None, latent_dim))
gen1 = generator(gen_input)

sess = tf.Session()
saver = tf.train.Saver()
try:
	saver.restore(sess, weight_path)
	print("[+] Weights loaded.")
except:
	print("[*] Weights couldn't load. Exiting.")
	exit()

def concat_images(X):
	outConcatImage = np.zeros(
		(
			int(math.sqrt(X.shape[0])) * X.shape[1],
			int(math.sqrt(X.shape[0])) * X.shape[1]
		)
	)
	for i in range(int(math.sqrt(X.shape[0]))):
		for j in range(int(math.sqrt(X.shape[0]))):
			x, y = i*X.shape[1], j*X.shape[1]
			outConcatImage[x:x+X.shape[1], y:y+X.shape[1]] = X[i*int(math.sqrt(X.shape[0]))+j]
	return outConcatImage

for episode_o in range(0, generate_count):
	noise_out = np.random.normal(size=(example_count, latent_dim))
	outImg = np.array(
		sess.run(
			gen1,
			feed_dict={
				gen_input:noise_out
			}
		) * 255.0,
		np.int32
	)
	cv2.imwrite("outs_generate/"+str(episode_o)+".jpg", concat_images(outImg))