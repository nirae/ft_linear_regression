# ft_linear_regression

> Le but de ce projet est de vous faire une introduction au concept de base du machine learning. Pour ce projet vous devrez créer un programme qui predit le prix d une voiture en utilisant l’entrainement par fonction linéaire avec un algorithme du gradient.

We have a dataset (data.csv) with some data on cars price/kilometers. We have to write a program to predict the car price with an input kilometers

## Training

```
$ ./train.py -h
usage: train.py [-h] [-i INPUT] [-o OUTPUT] [-H] [-s] [-l LEARNINGRATE] [-r RANGE]

Linear regression train program, using gradient descent

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input data file
  -o OUTPUT, --output OUTPUT
                        output theta file
  -H, --history         show history plot
  -s, --show            show regression plot
  -l LEARNINGRATE, --learningrate LEARNINGRATE
                        learning rate for training
  -r RANGE, --range RANGE
                        training range for training (epochs)
```

## Predict

```
$ ./predict.py -h
usage: predict.py [-h] [-d DATAFILE] [-t THETAFILE] [km]

Linear regression predict program, using a theta csv file, generated with the train program

positional arguments:
  km                    the kilometers of the car you want to predict the price

optional arguments:
  -h, --help            show this help message and exit
  -d DATAFILE, --datafile DATAFILE
                        input data file
  -t THETAFILE, --thetafile THETAFILE
                        output theta file
```

## Verify

```
./check_precision.py -h 
usage: check_precision.py [-h] [-d DATAFILE] [-t THETA]

Precision checker for ft_linear_regression predictions, using the R²-score formula

optional arguments:
  -h, --help            show this help message and exit
  -d DATAFILE, --datafile DATAFILE
                        input data file
  -t THETA, --theta THETA
                        input theta file
```
