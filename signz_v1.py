""" 
This Artificial Neural Network uses TF tensors only to classify
6 differents signs made using one hand: 0, 1, 2, 3, 4, 5 
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import contrib.coursera as c
import utils as utils

def initialize_params( layers_dim):
    L = len(layers_dim)

    params = {}
    for l in range(1,L):
        params["W" + str(l)] = tf.get_variable("W" + str(l), [layers_dim[l], layers_dim[l-1]], initializer = tf.contrib.layers.xavier_initializer())
        params["b" + str(l)] = tf.get_variable("b" + str(l), [layers_dim[l], 1], initializer = tf.zeros_initializer())

    return params


def forward(X, params):
    L = len(params) // 2
    A_prev = X
    for l in range(1,L+1):
        Z = tf.add( tf.matmul( params["W"+str(l)], A_prev), params["b"+str(l)])
        A_prev = tf.nn.relu( Z)

    return Z



def compute_cost(Z, Y, params, lambd = 0):
    # noTE: logits's shape should be (m, num_classes)
    logits = tf.transpose( Z)
    labels = tf.transpose( Y) 

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = labels))

    if ( 0 < lambd):
        l2_reg = 0
        L = len(params) // 2
        for l in range(1, L+1):
            l2_reg += tf.nn.l2_loss( params["W"+str(l)])

        cost = cost + tf.reduce_mean( lambd * l2_reg)

    return cost


def model(X_train, Y_train, X_test, Y_test, hparams):
    print(">> X_train.shape: {}".format(X_train.shape))
    print(">> Y_train.shape: {}".format(Y_train.shape))

    print(">> Learning Rate: {}".format(hparams["learning_rate"]))
    print(">> Hidden Layers: {}".format(hparams["hidden_layers"]))
    print(">> Weight decay: {}".format(hparams["l2_reg"]))


    ops.reset_default_graph() # allows to run again the model

    tf.set_random_seed(1)

    n_x = X_train.shape[0]
    n_y = Y_train.shape[0]

    # Computation Graph -------------------------------
    X = tf.placeholder(tf.float32, shape = [n_x, None])
    Y = tf.placeholder(tf.float32, shape = [n_y, None])

    params = initialize_params( [n_x]+hparams["hidden_layers"])
    Z = forward(X, params)
    cost = compute_cost( Z, Y, params, lambd = hparams["l2_reg"])
    train = hparams["optimizer"].minimize( cost)
    #^

    costs = []
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())

        for epoch in range(hparams["nb_epochs"]):
            epoch_cost = 0

            mini_batches = c.random_mini_batches(X_train, Y_train, hparams["batch_size"])
            num_mini_batches = len(mini_batches)

            for mini_batch in mini_batches:
                mb_X, mb_Y = mini_batch

                _, mb_cost = sess.run([train, cost], feed_dict={X:mb_X, Y: mb_Y})

                epoch_cost += mb_cost / num_mini_batches

            if 0 == (epoch % 100):
                print(">> Epoch: {} - Cost: {}".format(epoch, epoch_cost))

            if 0 == (epoch % 5):
                costs.append( epoch_cost)

        print(">> Epoch: {} - Cost: {}".format(hparams["nb_epochs"], epoch_cost))

        # Calculate Model's Accuracy
        correct_prediction = tf.equal(tf.argmax(Z), tf.argmax(Y))

        acc = tf.reduce_mean( tf.cast(correct_prediction, "float"))

        print(">> Train acc: {}".format( sess.run(acc, feed_dict={X:X_train, Y:Y_train})))
        print(">> Test  acc: {}".format( sess.run(acc, feed_dict={X:X_test, Y:Y_test})))

        params = sess.run( params)

    return params, costs


def predict( X, params):
    X = X / 255.

    x = tf.placeholder( tf.float32, shape = [X.shape[0], 1])
    params = utils.params_to_tensor( params)
    Z = forward( x, params)

    with tf.Session() as sess:
        logits = sess.run( Z, feed_dict={x:X})

        pred = {
            "logits": logits,
            "class": np.squeeze( tf.argmax( logits).eval()),
            "probs": tf.nn.softmax( logits, dim=0).eval()
        }
    return pred


###############################################################################
#
# TRAINING
#

trainset_x_orig, trainset_y_orig, testset_x_orig, testset_y_orig, classes = \
    c.h5_load_dataset( "datasets/train_signs.h5", "datasets/test_signs.h5")

# Flatten
trainset_x = trainset_x_orig.reshape( (trainset_x_orig.shape[0],-1)).T
trainset_y = c.one_hot( trainset_y_orig, classes.shape[0])

testset_x = testset_x_orig.reshape( (testset_x_orig.shape[0],-1)).T
testset_y = c.one_hot(testset_y_orig, classes.shape[0])

# Normalize
trainset_x = trainset_x / 255.
testset_x  = testset_x / 255.

#Hyperparameters
hp = {}
hp["learning_rate"] = 0.0001
hp["hidden_layers"] = [ 100, 25, 12, 6]
hp["nb_epochs"] = 800
hp["batch_size"] = 64
hp["optimizer"] = tf.train.AdamOptimizer( learning_rate = hp["learning_rate"])
hp["l2_reg"] = 0.01
#^

params, costs = model(trainset_x, trainset_y, testset_x, testset_y, hparams = hp)

#utils.plot_cost( costs, hp["learning_rate"])

#
# PREDICT
#
image_path = "samples/one.jpg"
img, x = c.load_image_as_array( image_path)
pred = predict(x, params)

print(">> Ground truth: {}".format(image_path))
print(">> Prediction: {} ({:.3f})".format( pred["class"], pred["probs"][ pred["class"]][0]))

plt.imshow(img)
plt.show()
