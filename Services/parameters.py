
class Parameters:

    def __init__(self):

        # 数据记录相关的参数
        self.record_limit = 20000 # 数据库的记录最大数
        self.drop_policy = 0.3 # 超过record limit 时 去除30% 前面的部分
        self.id_limit = 50000 # id 为单增计数，超过需要重新清除计数值
        self.trucate = True # 允许 id 清楚标志位

        # 模型训练相关参数
        self.partial_trainsize = 400 # 允许的部分最小训练集大小
        self.max_trainsize = 20000 #全部的记录训练
        self.batch_size = 200 # train bacth_size
        self.train_log = "./logs/test_train.log" # 训练的log记录
        # 训练设置数据 为 old、all、new
        # self.train_old =
        # self.train_new =






