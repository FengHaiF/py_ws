# 提取工艺参数

## 1. 主要模块说明

- **数据库（sqlite3）**

  用于存取用于模型训练的数据，格式为： ` num,x2 ,x4 ,temp,y1,y2 `，有数据量最大存储限制。

- **神经网络**

  用于预测 `y1, y2 `，使用神经网络模型，将 ` num,x2 ,x4 ,temp` 做为输入，离线训练的模型存放在`ANN\ann.pkl`文件，ANN文件夹下的py文件分别为产生模型`Gen_ann_model.py`和调用模型测试`load_model.py`。 
  模型调用接口位于 `Regressor.py`的 ANN 类  

- **集成学习**

  用于预测 `y1, y2 `，使用集成学习模型，其思想是将多个模型的输出综合起来得出结论，这里使用加权平均的方法。将 ` num,x2 ,x4 ,temp` 做为输入，离线训练的模型存放在`Ensemble\ensemble_ml.pkl，Ensemble_ml_y2.pkl`文件，ANN文件夹下的py文件分别为产生模型`all_model.py`和调用模型测试`load_ensemble_model.py`。 
  模型调用接口位于 `Regressor.py`的 Ensemble 类 

- **优化算法**

  针对最佳工艺参数，即如何将`y1, y2`尽可能大，对应的`num,x2 ,x4 ,temp`，考虑到`num`为单独的离散变量，将num作为一个输入参数。
  针对两个目标的多目标优化问题，这里使用`NSGA-II` 算法用于求解最佳工艺参数。其主要代码位于`optimize`文件加内，调用接口为`opt.py / solve(regress,Num)` 返回对应的x,y 。
  
- **socket 客户端**

  将机器学习作为服务的响应，根据接收的参数来处理具体业务
  分成5类：`predict, record, optimize, train, clear`。


## 2. json格式说明

- **predict**

  模型预测指令
 
  接收字典型数据格式

  ```
  {
    'params': 'predict',
    'method': 'ann', # 或者'ensmeble'
    'data': { # id, data_dict
             '1': {'Num':1,'x2':20,'x4':23,'temp':45},
             '2': {'Num':2,'x2':10,'x4':13,'temp':35},
             '3': {'Num':3,'x2':40,'x4':15,'temp':55},
            }
  }
  ```
  返回字典格式
  ```
  {
     # id, data_dict
    '1': {'y1':1,'y2':20},
    '2': {'y1':2,'y2':10},
    '3': {'y1':3,'y2':40}, 
  }  
  ```

 - **record**
   
   记录数据

   接收字典型数据格式
   ```
   {
       'params': 'record',
       'data':{  # id, data_dict
               '1': {'Num':1,'x2':20,'x4':23,'temp':45,'y1':4,'y2':5},
               '2': {'Num':2,'x2':10,'x4':13,'temp':35,'y1':4,'y2':5},
               '3': {'Num':3,'x2':40,'x4':15,'temp':55,'y1':4,'y2':5},
              }
   }
   ```
   返回格式
   ```
   {
        'record':True, 
        'insert_len':3, # 当前插入数据个数
        'record_len':100, # 当前数据库记录数 
        'max_id': 100, # 数据库当前的最大id
        'record_limit':pa.record_limit # 数据库的记录数据最大个数，配置文件设置
    }
   ```
 - **optimize**
   
   接收数据格式
   ```
   {
     'params': 'optimize',
     'method': 'ann',#或 'ensemble'
     'Num': 1 ,# 2,3
   }
   ```
   返回数据格式
   ```
   {
    'solve':'ensemble', # 对应method
     'x':[x2,x4,temp], # 如 [23，45，66]
     'y1':y[0], # 6.6
     'y2':y[1] # 7
    
    }
   ```

- **train**

    接收数据格式
    ```
    {
      'params': 'train',
      'method': 'ann',#或 'ensemble'
      'train_coff': 1 # 一般整数，小于6，train_size = 400*train_coff 
    }
    ```
    返回数据格式
    ```
    {
        'train': True, 
        'table_len': tab_len.fetchone(), # 当前数据库数据个数
        'train_size':train_size,#训练样本个数  'min_trainsize':pa.min_trainsize,#最小训练样本个数  
        'batch_size':pa.batch_size, # batch_size
        'max_trainsize': pa.max_trainsize  # 最大训练样本个数  
    }
    ```

- **clear** 

  接收数据库，id清零

  接收数据格式
  ```
  {
      'params':'clear',
  }
  ```
  返回数据格式
  ```
  {
      'clear':True,
  }
  ```


## 3. 参考资料

<a href="https://www.runoob.com/python/python-tutorial.html">python教程</a>

 <a href="https://www.cnblogs.com/subconscious/p/5058741.html">神经网络</a>

<a href=“https://blog.csdn.net/zwqjoy/article/details/80431496”>集成学习</a>

<a href=“https://blog.csdn.net/google19890102/article/details/46507387”>机器学习模型加权融合</a>

