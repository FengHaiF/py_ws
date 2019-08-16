#[0]: import library
import numpy as np
import pandas as pd
#
from sklearn.linear_model import Lasso,LinearRegression,Ridge,ElasticNet,TheilSenRegressor,HuberRegressor,RANSACRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import itertools

#[1]: import data
# 读取文件
data = pd.read_csv('tiqu.csv')
data.head()
data.describe()
Y = data['y1']  # 只预测波美度
var_names = list(data.columns)
#[var_names.remove(x) for x in ['y1','y2']]
varnames = ['Num','x2','x4','temp']
X = data[['Num','x2','x4','temp']]
X_train, X_test, y_train, y_test = train_test_split(X,Y,\
                            test_size=0.2,random_state=44)

#[2]: all models for regression
regs = [
    ['Lasso',Lasso()],
    ['LinearRegression',LinearRegression()],
    ['Ridge',Ridge()],
    ['ElasticNet',ElasticNet()],
    ['TheilSenRegressor',TheilSenRegressor()],
    ['RANSACRegressor',RANSACRegressor()],
    ['HuberRegressor',HuberRegressor()],
    ['SVR',SVR(kernel='linear')],
    ['DecisionTreeRegressor',DecisionTreeRegressor()],
    ['ExtraTreeRegressor',ExtraTreeRegressor()],
    ['AdaBoostRegressor',AdaBoostRegressor(n_estimators=6)],
    ['ExtraTreesRegressor',ExtraTreesRegressor(n_estimators=6)],
    ['GradientBoostingRegressor',GradientBoostingRegressor(n_estimators=6)],
    ['RandomForestRegressor',RandomForestRegressor(n_estimators=6)],
    ['XGBRegressor',XGBRegressor(n_estimators=6,)],
]
#[3]: evaluate all models by score mse value
preds = []
for reg_name,reg in regs:
    print(reg_name)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    score = mean_absolute_error(y_test,y_pred)
    preds.append([reg_name,y_pred])

# 对模型做各种组合寻找最优的方案
final_results = []
for comb_length in range(1,len(regs)+1):
    print('Model Amount :',comb_length)
    results = []
    for comb in itertools.combinations(preds,comb_length):
        pred_sum = 0
        model_name = []
        for reg_name,pred in comb:
            pred_sum += pred
            model_name.append(reg_name)
        pred_sum /= comb_length
        model_name = '+'.join(model_name)
        score = mean_absolute_error(y_test,pred_sum)
        results.append([model_name,score])
    results = sorted(results,key=lambda x:x[1])
    for model_name,score in results:
        print(model_name,score)
    print()
    final_results.append(results[0])

# final result
final_results = sorted(final_results,key=lambda x:x[1])

#for model_name,score in final_results:
#    print(model_name,score)

#[print(b) for b in zip(itertools.count(),[a[0] for a in regs])]
# show best plant
print("the best model is :")
print(final_results[0])

