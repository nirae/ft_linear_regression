#! /usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import argparse as arg
from progress.bar import ChargingBar

from Normalizer import Normalizer


class Trainer(object):

    def __init__(self, datafile, outfile, theta=[0, 0], learning_rate=0.1, train_range=1000):
        self.t_history = []
        self.theta = theta
        self.datafile = datafile
        self.output = outfile
        self.data = pd.read_csv(datafile)
        self.learning_rate = learning_rate
        self.range = train_range
        norm = Normalizer(self.datafile)
        self.km = norm.normalize_km_list(self.data['km'])
        self.price = norm.normalize_price_list(self.data['price'])

    def predict(self, km):
        """
        Model function:
            f(x) = ax + b
        """
        return self.theta[0] + (self.theta[1] * km)

    def mean_squared_error(self, km, price):
        """
        Fonction Cout /
        Mean squared error function:
            m = len(a) = len(b)
            theta0 = j(a, b) = (1 / m) * (sum(f(a[i]) - b[i]))
            theta1 = j(a, b) = (1 / m) * (sum((f(a[i]) - b[i]) * a[i]))
        a = km
        b = price
        """
        sum_t_0 = 0
        sum_t_1 = 0
        m = len(km)
        for i in range(m):
            sum_t_0 += self.predict(km[i]) - price[i]
            sum_t_1 += (self.predict(km[i]) - price[i]) * km[i]
        return [(1 / m) * sum_t_0, (1 / m) * sum_t_1]

    def gradient_descent(self):
        theta = self.theta
        bar = ChargingBar('Training', max=self.range, suffix='%(percent)d%% - eta %(eta)ds')
        for _ in range(self.range):
            bar.next()
            d_theta = self.mean_squared_error(self.km, self.price)
            theta[0] -= (self.learning_rate * (d_theta[0]))
            theta[1] -= (self.learning_rate * (d_theta[1]))
            self.t_history.append([theta[0], theta[1]])
        bar.finish()
        self.theta = theta
        return True

    def train(self, show_history=False, show_regression=False):
        self.gradient_descent()
        if show_history:
            plt.plot(range(1000), self.t_history)
            plt.show()
        if show_regression:
            line_x = [min(self.km), max(self.km)]
            line_y = [self.predict(i) for i in line_x]
            plt.xlabel('km')
            plt.ylabel('price')
            plt.plot(self.km, self.price, 'ro')
            plt.plot(line_x, line_y, 'b')
            plt.show()
        return self.theta
    
    def save_theta(self):
        with open(self.output, 'w') as file:
            file.write("theta0,theta1\n%f,%f" % (self.theta[0], self.theta[1]))
            file.close()
        print("The result 'theta' was written in the file %s" % self.output)


def main(args):
    trainer = Trainer(args.input, args.output, learning_rate=args.learningrate, train_range=args.range)
    theta = trainer.train(show_history=args.history, show_regression=args.show)
    print("Theta0 = %f\nTheta1 = %f" % (theta[0], theta[1]))
    trainer.save_theta()


if __name__ == '__main__':
    parser = arg.ArgumentParser(description="Linear regression train program, using gradient descent")
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='data.csv',
        help="input data file"
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='theta.csv',
        help="output theta file"
    )
    parser.add_argument(
        '-H',
        '--history',
        action="store_true",
        default=False,
        help="show history plot"
    )
    parser.add_argument(
        '-s',
        '--show',
        action="store_true",
        default=False,
        help="show regression plot"
    )
    parser.add_argument(
        '-l',
        '--learningrate',
        type=float,
        default=0.1,
        help="learning rate for training"
    )
    parser.add_argument(
        '-r',
        '--range',
        type=int,
        default=1000,
        help="training range for training (epochs)"
    )
    args = parser.parse_args()
    main(args)
