import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def standardize(X):
    mean_values = np.mean(X, axis=0)
    stand_otkl = np.std(X, axis=0)
    X_masht = (X - mean_values) / stand_otkl
    return X_masht


class Neuron:
    def __init__(self, input_size):
        self.weights = np.random.normal(size=input_size)
        self.b = np.random.normal()

    def feedforward_sum(self, inputs):
        return np.dot(self.weights, inputs) + self.b

    def feedforward_out(self, inputs):
        return sigmoid(self.feedforward_sum(inputs))


class NeuralNetwork:
    def __init__(self, input_size):
        self.h1 = Neuron(input_size)
        self.h2 = Neuron(input_size)
        self.h3 = Neuron(input_size)
        self.o1 = Neuron(3)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward_out(x)
        out_h2 = self.h2.feedforward_out(x)
        out_h3 = self.h3.feedforward_out(x)
        return self.o1.feedforward_out(np.array([out_h1, out_h2, out_h3]))

    def train(self, data, all_y_true):
        learn_rate = 10
        epochs = 300
        losses = []

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_true):
                sum_h1 = self.h1.feedforward_sum(x)
                h1 = self.h1.feedforward_out(x)

                sum_h2 = self.h2.feedforward_sum(x)
                h2 = self.h2.feedforward_out(x)

                sum_h3 = self.h3.feedforward_sum(x)
                h3 = self.h3.feedforward_out(x)

                sum_o1 = self.o1.feedforward_sum(np.array([h1, h2, h3]))
                o1 = self.o1.feedforward_out(np.array([h1, h2, h3]))

                y_pred = o1
                d_l_d_ypred = -2 * (y_true - y_pred)

                d_ypred_d_h1 = self.o1.weights[0] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.o1.weights[1] * deriv_sigmoid(sum_o1)
                d_ypred_d_h3 = self.o1.weights[2] * deriv_sigmoid(sum_o1)
                d_ypred_d_w1 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w2 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_w3 = h3 * deriv_sigmoid(sum_o1)
                d_ypred_d_b = deriv_sigmoid(sum_o1)


                self.o1.weights[0] -= learn_rate * d_l_d_ypred * d_ypred_d_w1
                self.o1.weights[1] -= learn_rate * d_l_d_ypred * d_ypred_d_w2
                self.o1.weights[2] -= learn_rate * d_l_d_ypred * d_ypred_d_w3
                self.o1.b -= learn_rate * d_l_d_ypred * d_ypred_d_b


                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_b = deriv_sigmoid(sum_h1)


                self.h1.weights[0] -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.h1.weights[1] -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.h1.weights[2] -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w3
                self.h1.b -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b


                d_h2_d_w1 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w2 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w3 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_b = deriv_sigmoid(sum_h2)


                self.h2.weights[0] -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w1
                self.h2.weights[1] -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w2
                self.h2.weights[2] -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.h2.b -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b


                d_h3_d_w1 = x[0] * deriv_sigmoid(sum_h3)
                d_h3_d_w2 = x[1] * deriv_sigmoid(sum_h3)
                d_h3_d_w3 = x[2] * deriv_sigmoid(sum_h3)
                d_h3_d_b = deriv_sigmoid(sum_h3)

                self.h3.weights[0] -= learn_rate * d_l_d_ypred * d_ypred_d_h3 * d_h3_d_w1
                self.h3.weights[1] -= learn_rate * d_l_d_ypred * d_ypred_d_h3 * d_h3_d_w2
                self.h3.weights[2] -= learn_rate * d_l_d_ypred * d_ypred_d_h3 * d_h3_d_w3
                self.h3.b -= learn_rate * d_l_d_ypred * d_ypred_d_h3 * d_h3_d_b

            if epoch % 10 == 0:
                y_preds = np.array([self.feedforward(x) for x in data])
                loss = mse(all_y_true, y_preds)
                losses.append(loss)

        return losses


if __name__ == "__main__":
    data = pd.read_csv('pacy.txt', delimiter='\t')

    features = data.iloc[:, :3].values
    other_features = data.iloc[:, 3:].values

    masht_features = standardize(features)

    full_data = np.concatenate((masht_features, other_features), axis=1)
    full_y = data['price_eur'].values

    train_data = full_data[:970]
    all_y_true = full_y[:970]

    test_data = full_data[970:]
    test_y = full_y[970:]

    network = NeuralNetwork(input_size=train_data.shape[1])
    losses = network.train(train_data, all_y_true)

    max_price = 69555
    for d, y in zip(test_data, test_y):
        predicted_price = network.feedforward(d) * max_price
        print(predicted_price, y * max_price)

    plt.plot(losses)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()
