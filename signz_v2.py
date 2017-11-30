""" 
This Artificial Neural Network uses TF Model to classify
6 differents signs made using one hand: 0, 1, 2, 3, 4, 5 
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import contrib.coursera as c
import utils as utils

#tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_MODEL = True
HPARAMS = {
    "learning_rate": 0.0001,
    "hidden_layers": [ 100, 25, 12, 6],
    "nb_epochs": 800,
    "batch_size": 64,
    "l2_reg": 1.4
}

def nn_model_fn( features, labels, params, mode):
    layers_dim = params["hidden_layers"]

    L = len(layers_dim)

    dense = tf.layers.dense( features["x"], layers_dim[0], activation = tf.nn.relu, name = "layer_1")
    for l in range(1, L-1):
        dense = tf.layers.dense( dense, layers_dim[l], activation = tf.nn.relu, name = "layer_" + str(l+1))

    # Logits layer (softmax)
    logits = tf.layers.dense( dense, layers_dim[L-1], name = "layer_" + str(L))

    predictions = {
        "classes": tf.argmax( logits, axis=1),
        "probabilities": tf.nn.softmax( logits, name="softmax_tensor")
    }

    # PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)


    # TRAIN
    onehot_labels = tf.one_hot( indices = tf.cast(labels, tf.int32), depth = 6)
    reg_cost = HPARAMS["l2_reg"] * sum( tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=onehot_labels), name="cost_tensor") + reg_cost

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer( learning_rate = params["learning_rate"]).minimize( cost, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode = mode, loss = cost, train_op = train_op)

    # EVAL
    eval_metrics_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels, predictions = predictions["classes"])
    }
    return tf.estimator.EstimatorSpec( mode = mode, loss = cost, eval_metric_ops = eval_metrics_ops)


def main(unused_argv):
    # TRAINING
    # trainset_x_orig.shape: (1080, 64, 64, 3)
    # trainset_y_orig.shape: (1, 1080)
    trainset_x_orig, trainset_y_orig, testset_x_orig, testset_y_orig, classes = \
        c.h5_load_dataset( "datasets/train_signs.h5", "datasets/test_signs.h5")

    # Flatten
    trainset_x = trainset_x_orig.reshape( (trainset_x_orig.shape[0],-1))
    trainset_y = trainset_y_orig.T

    testset_x = testset_x_orig.reshape( (testset_x_orig.shape[0],-1))
    testset_y = testset_y_orig.T

    # Normalize
    trainset_x = trainset_x / 255.
    testset_x  = testset_x / 255.

    print(">> X_train.shape: {}".format(trainset_x.shape))
    print(">> Y_train.shape: {}".format(trainset_y.shape))

    model = tf.estimator.Estimator(
        model_fn=nn_model_fn, 
        params = HPARAMS,  
        model_dir="/tmp/signz_dense_model")


    tensors_to_log = {"Cost ": "cost_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter = 100)

    if ( True == TRAIN_MODEL):
        model.train(input_fn = 
            tf.estimator.inputs.numpy_input_fn(
                x = {"x": trainset_x},
                y = trainset_y,
                batch_size = HPARAMS["batch_size"],
                num_epochs = HPARAMS["nb_epochs"],
                shuffle = True))


        # EVALUATION (train and test sets)
        train_metrics = model.evaluate(
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"x": trainset_x},
                y = trainset_y,
                batch_size = HPARAMS["batch_size"],
                num_epochs = 1,
                shuffle = False))
        print(">> Train metrics: %r"% train_metrics)

        test_metrics = model.evaluate(
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {"x": testset_x},
                y = testset_y,
                batch_size = HPARAMS["batch_size"],
                num_epochs = 1,
                shuffle = False))
        print(">> Test metrics: %r"% test_metrics)


    # PREDICT
    image_path = "samples/three_2.jpg"
    img, x = c.load_image_as_array( image_path)
    # x.shape: (12288, 1) 
    x = x.T / 255.

    predictions = model.predict(
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x": x},
            num_epochs = 1,
            shuffle = False))

    print(">> Ground truth: {}".format( image_path))
    for i, p in enumerate(predictions):
        print("Prediction: {} ({:.3f})".format(p["classes"], p["probabilities"][p["classes"]]))

    plt.imshow(img)
    plt.show()

    print(">> Done!")


if __name__ == "__main__":
    tf.app.run()
