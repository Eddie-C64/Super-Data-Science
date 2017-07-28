# This file creates a Restricted Boltzmann Machine.
# This file only runs of python2.7 via anaconda
# Part 1 - Data Processing
import torch
import pandas as pd
import torch.nn.parallel
import torch.utils.data
import numpy as np


movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

training_set = pd.read_csv('ml-100k/u1.base', sep="\t")
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', sep="\t")
test_set = np.array(test_set, dtype='int')

# Part 2 - Preparing Variable for Boltzmann Machine
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


def convert(data):

    new_data = []

    for id_users in range(1, nb_users):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_rating = data[:, 2][data[:, 0] == id_users]
        movie_ratings = np.zeros(nb_movies)
        movie_ratings[id_movies - 1] = id_rating
        new_data.append(list(movie_ratings))

    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert Data to PyTorch Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1
# Part 4 - Create the Architecture of the Restricted Boltzmann Machine


class RBM(object):
    def __init__(self, num_visible, num_hidden):
        self.Weights = torch.randn(num_visible, num_hidden)
        self.hidden_layer = torch.randn(1, num_hidden)  # Create single hidden layer
        self.visible_layer = torch.randn(1, num_visible)  # Only a single visible layer

    def sample_hidden(self, x):
        wx = torch.mm(x, self.Weights.t())
        activation = wx + self.hidden_layer.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_visible(self, x):
        wy = torch.mm(x, self.Weights)
        activation = wy + self.visible_layer.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v_zero, v_k_th, prob_h_vec_0, prob_h_vec_k):
        self.Weights += torch.mm(v_zero.t(), prob_h_vec_0) - torch.mm(v_k_th.t(), prob_h_vec_k)
        self.visible_layer += torch.sum((v_zero - v_k_th), 0)
        self.hidden_layer += torch.sum((prob_h_vec_0 - prob_h_vec_k), 0)

# Parameters
nv = len(training_set[0])  # The number is chosen by how many columns for data
nh = 100  # len(movies[0]) is too large, but sample to limit 100 features
batch_size = 100
nb_epoch = 10

rbm = RBM(nv, nh)

# Part 5 - Train the model
for epoch in range(1, nb_epoch + 1):
    training_loss = 0.0
    scaler_value = 0.0
    for id_user in range(0, nb_users-batch_size, 100):
        vk = training_set[id_user:id_user+batch_size]  # Kth node
        v0 = training_set[id_user:id_user+batch_size]  # visible node of kth position
        ph0, _ = RBM.sample_hidden(v0)

        # Apply Random Walk of a Monte Carlo Technique
        for k in range(nb_epoch):
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
            vk[v0 < 0] = v0[v0 < 0]

        phk, _ = rbm.sample_hidden(vk)
        rbm.train(v0, vk, ph0, phk)
        training_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        scaler_value += 1.
        print "epoch: " + str(epoch) + " loss: " + str(training_loss/scaler_value)

# Part 6 - Testing the Training Set
test_loss = 0.0
test_s = 0.0
for id_user in range(nb_users):
    v = test_set[id_user:id_user+batch_size]  # Kth node
    vt = training_set[id_user:id_user+batch_size]  # visible node target position

    # Apply Blind Walk of a Monte Carlo Technique
    if len(vt[vt >= 0]) > 0:
        _, ht = rbm.sample_hidden(vt)
        _, vt = rbm.sample_visible(ht)

        test_loss += torch.mean(torch.abs(v0[v0 >= 0] - vt[v0 >= 0]))
        test_s += 1.0
print "Test Loss: " + str(test_loss)
