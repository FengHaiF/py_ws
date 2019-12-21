import pickle
"""实际调用predict只需要用到pickle
   使用 y = predict(x) 进行神经网络预测;
   如果需要test 需要包含下面几个库"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# from pyecharts import options as opts
# from pyecharts.charts import Line
import numpy as np

def predict(X_test,file_path='ann_pipline.pkl'):
    """ X_test 的特征为 ['Num','x2','x4','temp']，
        本函数实现X_test的归一化到预测输出过程"""
    ann_piplines_dict = pickle.load(open(file_path,'rb'))
    # {"scaler_X":scaler_X,"ann":regress_net,"scaler_Y":scaler_Y}

    x_test = ann_piplines_dict["scaler_X"].transform(X_test)
    y_predict = ann_piplines_dict["ann"].predict(x_test)

    # 返回神经网络预测值
    return ann_piplines_dict["scaler_Y"].inverse_transform(y_predict)


def test(file_path='./html/line.html'):
    data = pd.read_csv('tiqu.csv')
    Y = data[['y1']].values
    X = data[['Num', 'x2', 'x4', 'temp']].values
    x_train, x_test, Y_train, Y_test = train_test_split(X, Y,\
                                                        test_size=0.2, random_state=44)
    y_predict_test =  predict(x_test)
    print("predict: ",y_predict_test.tolist())
    print("test: ",np.squeeze(Y_test).tolist())
    # test:  [4.5, 6.0, 6.5, 5.5, 5.5, 4.5, 4.5, 4.5, 5.5, 6.0, 6.5, 6.0, 6.0, 6.5]
    # error = mean_absolute_error(y_predict_test,Y_test)
    # # print("test [y1]:  ", error)
    # # test [y1]:   0.14596948557726444
    # y_predict_train = predict(x_train)
    # error = mean_absolute_error(y_predict_train, Y_train)
    # print("train [y1]:  ", error)
    # train [y1]:   0.21685998929835576
    # line = Line()
    #
    # line.add_xaxis(range(1, len(Y_test) + 1))
    # line.add_yaxis('samples', Y_test.reshape(-1, ),\
    #                label_opts=opts.LabelOpts(is_show=False))
    # line.add_yaxis('predict', y_predict_test,\
    #                label_opts=opts.LabelOpts(is_show=False))
    # line.set_global_opts(title_opts=opts.TitleOpts(title="line demo"))
    # line.render(file_path)



if __name__=="__main__":
    test()



