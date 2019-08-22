import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn import preprocessing as pre
import pandas as pd  
import matplotlib.pyplot as plt 
from datetime import datetime
def split_data(filename):
    data = pd.read_csv(filename)
    df = pd.DataFrame(data.values)
    df.columns = ['Datetime', 'Pageview']
    # print(df.Datetime)
    le_color = LabelEncoder()
    # print(df.Datetime)
    times = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df.values[:, 0]]
    X = [[time.day, time.hour, time.minute, time.second] for time in times]
    X = np.stack(X)
    y = df.values[:, 1]
    train_rate = 0.9
    X_train = X[:int(len(X) * train_rate)]
    y_train = y[:int(len(X) * train_rate)]
    X_test = X[int(len(X) * train_rate):]
    y_test = y[int(len(X) * train_rate):]
    # print(X_train)
    print('data has loaded and splited!!')

    return X_train, y_train, X_test, y_test
    

def train(filename):
    X_train, y_train, X_test, y_test = split_data(filename)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    # X_train = np.expand_dims(X_train, axis = 1)
    # X_train = scaler.fit_train
    # print(X_train)
    # X_test = np.expand_dims(X_test, axis = 1)
    reg_mlp = MLPRegressor(
    hidden_layer_sizes=(250),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=500, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # reg_svr = SVR(gamma = 'scale')
    # reg_svr.fit(X_train, y_train)

    reg_RFR = RandomForestRegressor(n_estimators = 50)
    # reg_RFR.fit(X_train, y_train)
    reg_GBR = GradientBoostingRegressor()
    reg_mlp.fit(X_train, y_train)

    y_predict = reg_mlp.predict(X_test)
    plt.plot(y_test)
    plt.plot(y_predict)
    plt.show()


if __name__ == '__main__':
    train('pageview_minute.csv')
    # print(convert('2019-06-13 15:00:00'))