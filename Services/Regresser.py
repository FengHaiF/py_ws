import pickle
"""实际调用predict只需要用到pickle
   使用 y = predict(x) 进行神经网络预测;
   如果需要test 需要包含下面几个库"""
# import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings(action='ignore', \
             module='sklearn')
             
class Ensemble(object):

    def __init__(self):
        self.models = {}
        self.weights = {}
    
    def load_model(self,folder_path='Ensemble',model_name='ensemble_ml'):
        y1_filepath = folder_path +'/'+model_name +'.pkl'
        models = pickle.load(open(y1_filepath,'rb')) # {names:model}
        self.models['y1'] = models
        self.weights['y1'] = [0.43,0.57]
        # y2 
        y2_filepath = folder_path +'/'+model_name +'_y2.pkl'
        models = pickle.load(open(y2_filepath,'rb')) # {names:model}
        self.models['y2']= models
        self.weights['y2'] = [0.2,0.2,0.2,0.2,0.2]
        print("load models from files using pickle")


    def predict(self,X_test):
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
        # models = pickle.load(open(file_path,'rb')) # {names:model}
        y_predicts = {}
        models = self.models['y1']
        weights = self.weights['y1']
        # y_predict_detail = []
        y_predict = 0
        for i, model in enumerate(models.values()):
            y_predict_temp = model.predict(X_test)
            y_predict += weights[i] * y_predict_temp
            # y_predict_detail.append(y_predict_temp)
        y_predicts['y1'] = y_predict[0]

        models = self.models['y2']
        weights = self.weights['y2']
        # y_predict_detail = []
        y_predict = 0
        for i, model in enumerate(models.values()):
            y_predict_temp = model.predict(X_test)
            y_predict += weights[i] * y_predict_temp
            # y_predict_detail.append(y_predict_temp)
        y_predicts['y2'] = y_predict[0]

        return y_predicts

    def update_model(self,cursor,batch_size):
        """
        :param cursor: 数据库查询
        :param batch_size: 训练的最小批大小
        :return:
        """

        # TODO: directly fit model from sql
        failed = False
        fail_reason = ""

        try:
            # conn = sqlite3.connect(db_path)
            # sqlQuery = conn.cursor()
            # 获取所有评论
            # sqlQuery.execute("SELECT * FROM  records")
            results = cursor.fetchmany(batch_size)

            y1_model = self.models['y1']
            y2_model = self.models['y2']

            print("update model...")
            while results:
                data = np.array(results)
                X_train = data[:,1:5]
                y1_model.partial_fit(X_train, data[:, 5])
                y2_model.partial_fit(X_train, data[:, 6])
                results = cursor.fetchmany(batch_size)

            y1_pkl_filename = 'Ensemble/ensemble_ml.pkl'
            pickle.dump(y1_model, open(y1_pkl_filename, 'wb'))

            y2_pkl_filename = 'Ensemble/ensemble_ml_y2.pkl'
            pickle.dump(y2_model, open(y2_pkl_filename, 'wb'))

            # update temp model
            new_model = {'y1': y1_model, 'y2': y2_model}
            self.models.update(new_model)
        except  Exception as e:
            failed = True
            fail_reason += e
            print('upate model failed...:',e)

        print("update model finished !!!")

        return failed, fail_reason





