import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import pickle

"""训练神经网络的模型，需要用到数据和一些库，
    生成'ann_pipline.pkl'文件用于固化模型"""

data = pd.read_csv('tiqu.csv')

Y = data[['y1']].values
X = data[['Num','x2','x4','temp']].values
x_train, x_test, Y_train, Y_test = train_test_split(X,Y,\
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
scaler_Y = StandardScaler().fit(Y_train)
y_train = scaler_Y.transform(Y_train)

#  训练神经网络
# 设置神经网络参数
regress_net = MLPRegressor(hidden_layer_sizes=(6,),\
                           activation="tanh",max_iter=500)
# 训练模型
regress_net.fit(x_train,y_train)
y_predict = regress_net.predict(x_test)
# error
y_predict = scaler_Y.inverse_transform(y_predict)

error = mean_absolute_error(y_predict,Y_test)
print("标准化，全部特征的神经网络训练结果")
print("test [y1,y2]:  ",error)
y_train_predict = regress_net.predict(x_train)
y_train_predict = scaler_Y.inverse_transform(y_train_predict)
error = mean_absolute_error(y_train_predict,Y_train)
print("train [y1,y2]:  ",error)


ann_pipline ={"scaler_X":scaler_X,"ann":regress_net,"scaler_Y":scaler_Y}

pickle.dump(ann_pipline,open("ann_pipline.pkl","wb"),-1)