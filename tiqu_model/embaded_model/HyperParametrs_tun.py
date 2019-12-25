import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle

"""训练神经网络的模型，需要用到数据和一些库，
    生成'ann_pipline.pkl'文件用于固化模型"""

data = pd.read_csv('tiqu.csv')

Y = data[['y1','y2']].values
X = data[['Num','x2','x4','temp']].values
x_train, x_test, y_train, y_test = train_test_split(X,Y,\
                            test_size=0.2,random_state=44)

# ann_pipline = make_pipeline(StandardScaler(),\
#                             MLPRegressor(hidden_layer_sizes=(6,),\
#                                     activation="tanh",max_iter=500))
# ann_pipline.fit_transform(x_train,y_test)
#
# y_predict = ann_pipline.predict(x_test)
#
# error = mean_absolute_error(y_predict,y_test)
# print("标准化，全部特征的神经网络训练结果")
# print("test [y1,y2]:  ",error)

# 标准化数据
scaler_X = StandardScaler().fit(x_train)
x_train = scaler_X.transform(x_train)
x_test = scaler_X.transform(x_test)
# y ----
# scaler_Y = StandardScaler().fit(Y_train)

regress_net = MLPRegressor(max_iter=10000)
tunned_parameters = {'hidden_layer_sizes':[(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(20,),(30,),\
                                            (4,2),(5,2),(6,3),(7,3),(8,4),(9,4),(10,5),(15,7),(20,8),(25,9),(30,10) ],\
                                'activation':['identity', 'logistic', 'tanh', 'relu'],\
                                 'solver': ['lbfgs', 'sgd', 'adam'],\
                                'learning_rate':['constant', 'invscaling', 'adaptive']}

# print(sklearn.metrics.SCORERS.keys())
"""
dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 
'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 
'accuracy', 'roc_auc', 'balanced_accuracy', 'average_precision', 'neg_log_loss', 
'brier_score_loss', 'adjusted_rand_score', 'homogeneity_score', 'completeness_score',
 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score',
  'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples',
   'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted',
    'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro', 
    'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'])
"""
regress = GridSearchCV(regress_net, tunned_parameters, cv=5,
                       scoring='neg_mean_absolute_error',n_jobs=4)

regress.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(regress.best_params_)
print()
print("Grid scores on development set:")
print()
means = regress.cv_results_['mean_test_score']
stds = regress.cv_results_['std_test_score']
#这里输出了各种参数在使用交叉验证的时候得分的均值和方差

for mean, std, params in zip(means, stds, regress.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()
#  训练神经网络
# 设置神经网络参数
# regress_net = MLPRegressor(hidden_layer_sizes=(8,5),\
                        #    activation="relu",max_iter=8000)
# 训练模型
# regress_net.fit(x_train,y_train)
# y_predict = regress_net.predict(x_test)
# # error
# print("y_predict :",y_predict[2],"|| y_test:",y_test[2])

# error = mean_absolute_error(y_predict,y_test)
# print("标准化，全部特征的神经网络训练结果")
# print("test [y1,y2]:  ",error)
# y_train_predict = regress_net.predict(x_train)

# error = mean_absolute_error(y_train_predict,y_train)
# print("train [y1,y2]:  ",error)


# ann_pipline ={"scaler_X":scaler_X,"ann":regress_net}

# pickle.dump(ann_pipline,open("ann.pkl","wb"),-1)