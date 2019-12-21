import socket
import json
import pandas as pd 
from  numpy import random
import time
HOST='localhost'
PORT=10001
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)      #定义socket类型，网络通信，TCP
s.connect((HOST,PORT))       #要连接的IP与端口


data = pd.read_csv('Ensemble/tiqu.csv')
# response = s.recv(10240)  # 把接收的数据定义为变量
# sendData ={
#      '2016-07-21':{
#          'value':3934,
#          'titles':[u'标题1',u'标题2',u'标题3']
#      },
#      '2016-07-22':{
#          'value':1109,
#          'titles':[u'标题4',u'标题5',u'标题6']
#      },
#     '2016-07-23':{
#          'value':2365,
#          'titles':[u'标题7',u'标题8',u'标题9']
#      }
# }

def load_test_data(file_path = 'Ensemble/tiqu.csv'):
    # data = pd.read_csv(file_path)
    #Y = data[['y1']].values
    index = random.randint(0,50,size=(5,))
    #index = [0,1,2,3,4]
    x_table = data.loc[:,['Num', 'x2', 'x4', 'temp']]
    y_table = data.loc[:,['y1', 'y2']]
    X = x_table.iloc[index]

    Y = y_table.iloc[index]

    return X,Y


def Client(sendData):
    message = json.dumps(sendData)  #json 经过转换后才能传输
    s.sendall(message)      #把命令发送给对端


    response=s.recv(10240)     #把接收的数据定义为变量
    jresp = json.loads(response)

    print("Recv",jresp)         #输出返回的json


def test():
    X, Y = load_test_data()
    message = X.to_json(orient='index')
    print(message)
    s.sendall(message.encode('utf-8'))  # 把命令发送给对端
    print("sucess")
    print("Y = ", Y)

    response = s.recv(10240)  # 把接收的数据定义为变量
    jresp = json.loads(response.decode('utf-8'))

    print("Recv", jresp)  # 输出返回的json
    print("------------")

def main():
    test()
    print("------------")
    time.sleep(3)
    test()
    print("------------")
    time.sleep(4)
    test()
    print("------------")

    # print("Y:",Y)  


if __name__ =="__main__":
    main()