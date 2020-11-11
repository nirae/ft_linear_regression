#! /usr/bin/env python3

import pandas as pd
import argparse as arg

from Normalizer import Normalizer

class Predicter(object):

    def __init__(self, thetafile='theta.csv', datafile='data.csv'):
        try:
            df = pd.read_csv(thetafile)
            self.theta = [
                float(df['theta0'].astype(float)),
                float(df['theta1'].astype(float))
            ]
        except:
            print("the theta file was not found")
            self.theta = None
        self.norm = Normalizer(datafile)

    def predict(self, km):
        """
        Model function:
            f(x) = ax + b
        """
        return self.theta[0] + (self.theta[1] * km)

    def run(self, km):
        if self.theta == None:
            return None
        norm_km = self.norm.normalize_km(km)
        result = self.predict(norm_km)
        return self.norm.denormalize_price(result)


def main(args):
    predicter = Predicter(datafile=args.datafile, thetafile=args.thetafile)
    result = predicter.run(args.km)
    if not result:
        result = 0
    result = round(result, 2)
    print(result)

if __name__ == '__main__':
    parser = arg.ArgumentParser(description="Linear regression predict program, using a theta csv file, generated with the train program")
    parser.add_argument(
        '-d',
        '--datafile',
        type=str,
        default='data.csv',
        help="input data file"
    )
    parser.add_argument(
        '-t',
        '--thetafile',
        type=str,
        default='theta.csv',
        help="output theta file"
    )
    parser.add_argument(
        'km',
        type=int,
        help="the kilometers of the car you want to predict the price"
    )
    args = parser.parse_args()
    main(args)
