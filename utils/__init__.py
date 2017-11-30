import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def params_to_tensor( params):
    L = len(params) // 2

    ptensors = {}
    for l in range(1, L+1):
        ptensors["W"+str(l)] = tf.convert_to_tensor(params["W"+str(l)])
        ptensors["b"+str(l)] = tf.convert_to_tensor(params["b"+str(l)])

    return ptensors


def plot_cost( costs, lr):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(lr))
    plt.show()