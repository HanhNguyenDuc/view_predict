import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd  
import matplotlib.pyplot as plt 

def convert(d):
    splited = d.split(' ')
    time = splited[1].split(':')
    dates = splited[0].split('-')
    time = list(map(int, time))
    return time
    

def split_data(filename):
    data = pd.read_csv(filename)
    # print(data[0])
    values = data.values
    X = values[:,0]
    y = values[:,1]
    X = [convert(d) for d in X]
    X_test = X[:int(len(X) * 0.2)]
    y_test = y[:int(len(y) * 0.2)]
    X_val = X[int(len(X) * 0.2):int(len(X) * 0.3)]
    y_val = y[int(len(X) * 0.2):int(len(X) * 0.3)]
    X_train = X[int(len(X) * 0.3):]
    y_train = y[int(len(X) * 0.3):]
    return (X_train, y_train, X_test, y_test, X_val, y_val)

def train(filename):
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(filename)
    poly_reg = PolynomialFeatures(degree = 5)
    X_train = poly_reg.fit_transform(X_train)
    X_test = poly_reg.fit_transform(X_test)
    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_predict = reg.predict(X_test)
    plt.plot(y_test)
    plt.plot(y_predict)
    plt.show()


if __name__ == '__main__':
    train('pageview_minute.csv')
    # print(convert('2019-06-13 15:00:00'))