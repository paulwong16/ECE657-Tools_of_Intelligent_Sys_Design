from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy
import numpy as np


if __name__ == '__main__':
    y_pred = test_mlp('./train_data.csv')

    test_labels = np.genfromtxt('./train_labels.csv', delimiter=',')
    test_labels = test_labels

    test_accuracy = accuracy(test_labels, y_pred) * 100

    print(test_accuracy)
