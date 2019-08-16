import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
from collections import deque


# 读取文件
def net_eval(data,plot_fig=False):
    #data = pd.read_csv('tiqu.csv')

    Y = data[['y1']].values
    #var_names = list(data.columns)
    #[var_names.remove(x) for x in ['y1','y2']]
    #varnames = ['Num','x2','x4','temp']
    X = data[['Num','x2','x4','temp']].values
    x_train, x_test, train_Y, test_Y = train_test_split(X,Y,\
                                test_size=0.2,random_state=44)
    ## 标准化数据

    # y ----
    scaler_Y = StandardScaler().fit(train_Y)
    y_train = scaler_Y.transform(train_Y)
    y_test = scaler_Y.transform(test_Y)

    #  训练神经网络
    # 设置神经网络参数
    regress_net = MLPRegressor(hidden_layer_sizes=(6,),\
                               activation="tanh",max_iter=500)
    # 训练模型
    regress_net.fit(x_train,y_train)
    y_predict = regress_net.predict(x_test)
    # error
    y_predict = scaler_Y.inverse_transform(y_predict)

    error = mean_absolute_error(y_predict,test_Y)
    print("标准化，全部特征的神经网络训练结果")
    print("test [y1,y2]:  ",error)
    y_train_predict = regress_net.predict(x_train)
    y_train_predict = scaler_Y.inverse_transform(y_train_predict)
    error = mean_absolute_error(y_train_predict,train_Y)
    print("train [y1,y2]:  ",error)
    if plot_fig:
        plt.plot(y_predict)
        plt.plot(test_Y,'*')
        plt.ylabel("波美度")
        plt.legend(["预测值","实际值"])
        plt.show()
    return (regress_net,scaler_Y)



# 可视化神经网络模型

# 画3维曲面
'''
def Show_3D(plot_fig=False):
    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)
    X,Y = np.meshgrid(x,y)
    Z = np.zeros((100,100))
    inputs = np.array([1.,1.,1.,75.]).reshape((1,-1))
    inputs = inputs

    for i in range(0,100):
        for j in range(0,100):
            inputs[:,1:-1] = x[i],y[j]
            Z[i,j] = scaler_Y.inverse_transform(regress_net.predict(inputs))
    if plot_fig:
        fig = plt.figure()
        axes3d = Axes3D(fig)
        axes3d.plot_surface(X,Y,Z)
        axes3d.scatter3D(x_test[:,1],x_test[:,2],y_predict.reshape((-1,1)),c='red')
        plt.xlabel("x2  scale to [-1,1]")
        plt.ylabel("x4 scale to [-1,1]")
        plt.title("x2、x4的拟合曲面")
        plt.show()
    return {'surf':(X,Y,Z),'scatter':(x_test[:,1],x_test[:,2],y_predict.reshape((-1,1))) }
'''

if __name__=="__main__":
    data = pd.read_csv('tiqu.csv')
    net_eval(data, plot_fig=True)