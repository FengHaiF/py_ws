import numpy as np
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line,Surface3D
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def net_eval(data):
    #data = pd.read_csv('tiqu.csv')

    Y = data[['y1']].values
    X = data[['Num','x2','x4','temp']].values
    x_train, x_test, train_Y, test_Y = train_test_split(X,Y,\
                                test_size=0.2,random_state=44)
    ## 标准化数据
    scaler_X = StandardScaler().fit(x_train)
    x_train = scaler_X.transform(x_train)
    x_test = scaler_X.transform(x_test)
    # y ----
    scaler_Y = StandardScaler().fit(train_Y)
    y_train = scaler_Y.transform(train_Y)

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
    return (regress_net,scaler_X,scaler_Y)

# 读取文件
data = pd.read_csv('tiqu.csv')
Y = data[['y1']].values
X = data[['Num','x2','x4','temp']].values
x_train, x_test, train_Y, test_Y = train_test_split(X,Y,\
                            test_size=0.2,random_state=44)
net,scaler_X,scaler_Y = net_eval(data)

y_predict = scaler_Y.inverse_transform(net.predict(\
                             scaler_X.transform(x_test)))
line = Line()

line.add_xaxis(range(1,len(test_Y)+1))
line.add_yaxis('samples',test_Y.reshape(-1,),\
               label_opts=opts.LabelOpts(is_show=False))
line.add_yaxis('predict',y_predict,\
               label_opts=opts.LabelOpts(is_show=False))
line.set_global_opts(title_opts=opts.TitleOpts(title="line demo"))
line.render('./html/line.html')

# 3D surface
target_feature = (1, 2)
pdp, axes = partial_dependence(net,\
                               scaler_X.transform(x_train),\
                               target_feature,\
                               grid_resolution=30)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].T

names = ['Num','x2','x4','temp']
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                       cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.suptitle('Partial dependence of house value on median\n'
             'age and average occupancy, with Gradient Boosting')
plt.subplots_adjust(top=0.9)

plt.show()

x = XX.reshape((-1,1))
y = YY.reshape((-1,1))
z = scaler_Y.inverse_transform(Z.reshape((-1,1)))
data_xyz = np.hstack((x,y,z)).tolist()
## pyecharts Surface3D

(

    Surface3D(init_opts=opts.InitOpts(width="1600px", height="800px"))
    .add(
        series_name="",
        shading="color",
        data=data_xyz,
        xaxis3d_opts=opts.Axis3DOpts(type_="value"),
        yaxis3d_opts=opts.Axis3DOpts(type_="value"),
        zaxis3d_opts=opts.Axis3DOpts(min_=4,max_=8),
        grid3d_opts=opts.Grid3DOpts(width=100, height=40, depth=100),
    )
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(
            dimension=2,
            max_=8,
            min_=4,
            range_color=[
                "#313695",
                "#4575b4",
                "#74add1",
                "#abd9e9",
                "#e0f3f8",
                "#ffffbf",
                "#fee090",
                "#fdae61",
                "#f46d43",
                "#d73027",
                "#a50026",
            ],
        )
    )
    .render("./html/net_surf3D.html")

)

#print("localhost:63342/py_ws/tiqu_model/pyecharts_view/html/net_surf3D.html")