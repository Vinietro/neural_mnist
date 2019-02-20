import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_train = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test = mnist.test.num_examples

learning_rate = 1e-4
n_iterations = 1000
batch_size = 128
dropout = 0.5

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
keep_prob = tf.placeholder(tf.float32)

#Definimos os pesos das conexões definindo o input e o output de cada uma
#Começamos com o input de 784(28x28px), e com 3 comadas ocultas, chegamos no output de 10 (0-9 digitos)
weights = {
    'w1': tf.Variable(tf.truncated_normal([784, 512], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([512, 256], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([256, 128], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([128, 10], stddev=0.1)),
}

#Definimios os pesos dos erros para correção e aprendizado da maquina
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[512])),
    'b2': tf.Variable(tf.constant(0.1, shape=[256])),
    'b3': tf.Variable(tf.constant(0.1, shape=[128])),
    'out': tf.Variable(tf.constant(0.1, shape=[10]))
}

#Adicionamos as layers a rede neural
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Chamamos o run() para treinar a rede de acordo com o tamanho do batch
for i in range(n_iterations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # Printamos na tela qual intereação foi e a taxa de perda e precisão
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Interacao", str(i), "\t| Perda =", str("{:10.2f}".format(minibatch_loss*100)), "%\t| Precisao =", str("{:10.2f}".format(minibatch_accuracy*100)), "%")

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
print("\nPrecisao no teste:","{:10.2f}".format(test_accuracy*100), "%")

#Adicionamos uma imagem nump e invertemos ela porque ele retorna um vetor de 256 tons de branco a preto ao contrário
img = np.invert(Image.open("test_img.png").convert('L')).ravel()

#Chamamos a rede para tentar prever a imagem que desenhamos
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Previsao da primeira imagem:", np.squeeze(prediction))

#Adicionamos uma imagem nump e invertemos ela porque ele retorna um vetor de 256 tons de branco a preto ao contrário
img = np.invert(Image.open("test_img_2.png").convert('L')).ravel()

#Chamamos a rede para tentar prever a imagem que desenhamos
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
print ("Previsao da segunda imagem:", np.squeeze(prediction))