import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from bayes_opt import BayesianOptimization

data = pd.read_csv('tiqu.csv')

Y = data[['y1','y2']].values
X = data[['Num','x2','x4','temp']].values
x_train, x_test, y_train, y_test = train_test_split(X,Y,\
                            test_size=0.2,random_state=44)


# 标准化数据
scaler_X = StandardScaler().fit(x_train)
x_train = scaler_X.transform(x_train)
x_test = scaler_X.transform(x_test)

# 由于bayes优化只能优化连续超参数，
def rf_cv(hidden_layer_sizes, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999), # float
            max_depth=int(max_depth),
            random_state=2
        ),
        x, y, scoring='neg_mean_absolute_error', cv=5
    ).mean()
    return val

parameters = {'hidden_layer_sizes':[(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(20,),(30,),\
                                            (4,2),(5,2),(6,3),(7,3),(8,4),(9,4),(10,5),(15,7),(20,8),(25,9),(30,10) ],\
                                'activation':['identity', 'logistic', 'tanh', 'relu'],\
                                 'solver': ['lbfgs', 'sgd', 'adam'],\
                                'learning_rate':['constant', 'invscaling', 'adaptive']}