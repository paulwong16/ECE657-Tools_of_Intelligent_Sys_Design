import numpy as np

from MLP import MLP

STUDENT_NAME = 'Zhijie Wang'
STUDENT_ID = '20856733'


def test_mlp(data_file):
    X = np.genfromtxt(data_file, delimiter=',')
    X = X.T

    mlp = MLP()
    mlp.read_model('./pretrained_model.pkl')
    y = mlp.predict(X)
    return y.T


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''
