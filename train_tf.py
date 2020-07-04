import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import cv2, math

example_count = 64
batch_size = 256
max_episode = 10000
episode_each = 100
latent_dim = 100

np.random.seed(0)
static_noise_out = np.random.normal(size=(example_count, latent_dim))

weight_path = "model/gan_mnist"

# Load train and test slices, merge them.
(x_train, _), (x_test, _) = mnist.load_data()
X = np.concatenate([x_train, x_test], axis=0)
np.random.shuffle(X)

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

# Discriminator.
def discriminator(inp, reuse):
	with tf.variable_scope("dis", reuse=reuse):
		inp = tf.expand_dims(inp, axis=-1)

		x1 = tf.layers.conv2d(inp, 16, kernel_size=5, strides=1, padding="valid")
		x1 = tf.nn.relu(x1)

		x2 = tf.layers.conv2d(x1, 32, kernel_size=5, strides=1, padding="valid")
		x2 = tf.layers.batch_normalization(x2, training=True)
		x2 = tf.nn.relu(x2)

		x3 = tf.layers.conv2d(x2, 64, kernel_size=5, strides=1, padding="valid")
		x3 = tf.layers.batch_normalization(x3, training=True)
		x3 = tf.nn.relu(x3)

		return tf.layers.dense(tf.layers.flatten(x3), 1, activation=None)

dis_real_input = tf.placeholder(tf.float32, shape=(None, 28, 28))
gen_input = tf.placeholder(tf.float32, shape=(None, latent_dim))

gen1 = generator(gen_input)

dis1 = discriminator(dis_real_input, False)
dis2 = discriminator(gen1, True)

# Losses.
gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(dis2, tf.float32), logits=dis2))
gen_train = tf.train.AdamOptimizer(0.0001).minimize(gen_loss, var_list=tf.trainable_variables("gen/"))

dis_loss = tf.reduce_mean(
	tf.add(
		tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(dis1, tf.float32), logits=dis1)),
		tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(dis2, tf.float32), logits=dis2))
	)
)
# Make discriminator slightly more powerful.
dis_train = tf.train.AdamOptimizer(0.0003).minimize(dis_loss, var_list=tf.trainable_variables("dis/"))

sess = tf.Session()
try:
	saver.restore(sess, weight_path)
	print("[+] Weights loaded.")
except:
	sess.run(tf.global_variables_initializer())
	print("[!] Weights couldnt load. Initialized.")
saver = tf.train.Saver()

# Generator class, each call returns a mini-batch from dataset.
class DataGenerator:
	def __init__(self):
		self.index = 0

	def __call__(self):
		if self.index+batch_size < X.shape[0]:
			xx = X[self.index:self.index+batch_size] / 255.0
			self.index += batch_size
			return xx
		else:
			xx = X[self.index:] / 255.0
			self.index = 0
			return xx

data_generate = DataGenerator()

# Takes images and merges into one image, for monitoring.
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

# Discriminator initial training.
for episode in range(0, episode_each*10):
	batch_noise = np.random.normal(size=(batch_size, latent_dim))
	batch_data = data_generate()

	sess.run(
		dis_train,
		feed_dict={
			dis_real_input:batch_data,
			gen_input:batch_noise
		}
	)

	if episode % 100 == 0:
		gl, dl = sess.run(
			[gen_loss, dis_loss],
			feed_dict={
				dis_real_input:batch_data,
				gen_input:batch_noise
			}
		)
		print("Dis Ep {}, Gen Loss {}, Dis Loss {}".format(episode, gl, dl))

for episode_o in range(0, max_episode):
	# Train discriminator.
	for episode in range(0, episode_each):
		batch_noise = np.random.normal(size=(batch_size, latent_dim))
		batch_data = data_generate()

		sess.run(
			dis_train,
			feed_dict={
				dis_real_input:batch_data,
				gen_input:batch_noise
			}
		)

	gl, dl = sess.run(
		[gen_loss, dis_loss],
		feed_dict={
			dis_real_input:batch_data,
			gen_input:batch_noise
		}
	)
	print("Ep {}, Dis Ep {}, Gen Loss {}, Dis Loss {}".format(episode_o, episode, gl, dl))

	# Train generator.
	for episode in range(0, episode_each):
		batch_noise = np.random.normal(size=(batch_size, latent_dim))
		sess.run(
			gen_train,
			feed_dict={
				gen_input:batch_noise
			}
		)

	batch_data = data_generate()
	gl, dl = sess.run(
		[gen_loss, dis_loss],
		feed_dict={
			dis_real_input:batch_data,
			gen_input:batch_noise
		}
	)
	print("Ep {}, Gen Ep {}, Gen Loss {}, Dis Loss {}".format(episode_o, episode, gl, dl))

	# Save sample image from generator.
	outImg = np.array(
		sess.run(
			gen1,
			feed_dict={
				gen_input:static_noise_out
			}
		) * 255.0,
		np.int32
	)
	cv2.imwrite("outs/"+str(episode_o)+".jpg", concat_images(outImg))

	# Save progress.
	if episode_o%100 == 0:
		saver.save(sess, weight_path, global_step=episode_o, write_meta_graph=False)

	print("[*] Episode {} finished.".format(episode_o))

# Final save.
saver.save(sess, weight_path+"_final", write_meta_graph=False)