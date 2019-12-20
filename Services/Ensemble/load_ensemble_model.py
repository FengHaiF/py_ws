import pickle
"""实际调用predict只需要用到pickle
   使用 y = predict(x) 进行神经网络预测;
   如果需要test 需要包含下面几个库"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from pyecharts import options as opts
from pyecharts.charts import Line
import numpy as np


def predict(X_test,weights=[0.43,0.57],file_path='ensemble_ml.pkl'):
    """
        ['RANSACRegressor+RandomForestRegressor', 0.08121470017053545]
        ['RANSACRegressor', 'RandomForestRegressor']
        RANSACRegressor  score: 0.2120331572321849
        RandomForestRegressor  score: 0.14285714285714285
        根据 score视情况设置合适的权重weights
        weights=[0.43,0.57]
        test [y1]:   0.07817797547999396
        train [y1]:   0.1510962099583446
    """
    models = pickle.load(open(file_path,'rb')) # {names:model}
    y_predict = 0
    y_predict_detail = []
    for i, model in enumerate(models.values()):
        y_predict_temp = model.predict(X_test)
        y_predict += weights[i] * y_predict_temp
        y_predict_detail.append(y_predict_temp)

    return y_predict,y_predict_detail



def test_withPlot(file_path='./html/line.html'):
    data = pd.read_csv('../tiqu.csv')
    Y = data[['y1']].values
    X = data[['Num', 'x2', 'x4', 'temp']].values
    x_train, x_test, Y_train, Y_test = train_test_split(X, Y,\
                                                        test_size=0.2, random_state=44)
    y_predict_test,y_predicts =  predict(x_test)
    error = mean_absolute_error(y_predict_test,Y_test)
    print("test [y1]:  ", error)
    # test [y1]:
    y_predict_train,_ = predict(x_train)
    error = mean_absolute_error(y_predict_train, Y_train)
    print("train [y1]:  ", error)
    # train [y1]:
    line = Line()

    line.add_xaxis(range(1, len(Y_test) + 1))
    line.add_yaxis('samples', Y_test.reshape(-1, ),\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('weighted_predict', y_predict_test,\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('RANSACRegressor', y_predicts[0],\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('RandomForestRegressor', y_predicts[1],\
                   label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(title_opts=opts.TitleOpts(title="ensemble test"))
    line.render(file_path)

def test_y2(file_path='./html/line_y2.html'):
    data = pd.read_csv('../tiqu.csv')
    Y = data[['y2']].values
    X = data[['Num', 'x2', 'x4', 'temp']].values
    x_train, x_test, Y_train, Y_test = train_test_split(X, Y,\
                                                        test_size=0.2, random_state=44)

    weights = [ 0.2,0.2,0.2,0.2,0.2]
    y_predict_test,y_predicts =  predict(x_test,weights,file_path='ensemble_ml_y2.pkl')
    error = mean_absolute_error(y_predict_test,Y_test)
    print("test [y2]:  ", error)

    
    y_predict_train,_  = predict(x_train,weights,file_path='ensemble_ml_y2.pkl')
    error = mean_absolute_error(y_predict_train, Y_train)
    print("train [y2]:  ", error)
    # train [y1]:
    line = Line()

    line.add_xaxis(range(1, len(Y_test) + 1))
    line.add_yaxis('samples', Y_test.reshape(-1, ),\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('weighted_predict', y_predict_test,\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('RANSACRegressor', y_predicts[0],\
                   label_opts=opts.LabelOpts(is_show=False))
    line.add_yaxis('RandomForestRegressor', y_predicts[1],\
                   label_opts=opts.LabelOpts(is_show=False))
    line.set_global_opts(title_opts=opts.TitleOpts(title="ensemble test"))
    line.render(file_path)

def Gen_FakeData():
    data = pd.read_csv('./Gendata.csv')
    X = data[['Num', 'x2', 'x4', 'temp']].values

    # y1 with noise 
    y1_predict , _ =  predict(X)
    y1_fake = y1_predict + 0.1*np.random.randn(len(X),1) 
    # y2 with noise
    weights = [ 0.2,0.2,0.2,0.2,0.2]
    y2_predict , _ =  predict(X,weights,file_path='ensemble_ml_y2.pkl')
    y2_fake = y2_predict + 0.3*np.random.randn(len(X),1) 
    
    data['y1'] = y1_fake[0]
    data['y2'] = y2_fake[0]
    data.to_csv("./Fake_gen.csv")
     




if __name__=="__main__":
    Gen_FakeData()