#! /usr/bin/env python3

import pandas as pd
import argparse as arg

from predict import Predicter

class Precision(object):

    def __init__(self, datafile='data.csv', thetafile='theta.csv'):
        self.datafile = datafile
        self.data = pd.read_csv(datafile)
        self.y = self.data['price']
        self.predictions = []
        predicter = Predicter(thetafile)
        for val in self.data['km']:
            result = predicter.run(val)
            self.predictions.append(result)

    def list_mean(self, values):
        r = 0
        for i in values:
            r += i
        return r / len(values)

    def verify(self):
        """
        Coefficient of determination or R2 score
        R2 = 1 - (sum((y[i] - values[i]) ** 2)) / (sum((y[i] - y_mean) ** 2))
        """
        u = 0
        for i in (self.y - self.predictions) ** 2:
            u += i
        v = 0
        for i in (self.y - self.list_mean(self.y)) **2:
            v += i
        return 1 - u / v

def main(args):
    prec = Precision(datafile=args.datafile, thetafile=args.theta)
    print("Some predictions for all values in the dataset will be done and compared with the real data, with a R2-score algorithm")
    print("Coefficient of determination", prec.verify())


if __name__ == '__main__':
    parser = arg.ArgumentParser(description="Precision checker for ft_linear_regression predictions, using the R²-score formula")
    parser.add_argument(
        '-d',
        '--datafile',
        type=str,
        default='data.csv',
        help="input data file"
    )
    parser.add_argument(
        '-t',
        '--theta',
        type=str,
        default='theta.csv',
        help="input theta file"
    )
    args = parser.parse_args()
    main(args)
