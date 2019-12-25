#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-
#
import socket
import threading
import socketserver
import sqlite3
import json, types,string
import os, time
import numpy as np
import warnings
from Regresser import Ensemble,ANN
from parameters import Parameters

from opt import solve
warnings.filterwarnings(action='ignore', \
             module='sklearn')

pa = Parameters()

# 导入ensemble 的模型
ensemble = Ensemble()
ensemble.load_model()

ann = ANN()
ann.load_model()

db_path = 'db/test_records.db'

def deleteOldRecord(cursor,drop_len):
    """去除旧的记录"""
    tab_len = cursor.execute("SELECT count(id) from RECORDS")
    tab_len = list(tab_len)[0][0]
    if tab_len > pa.record_limit:
        cursor.execute("DELETE FROM RECORDS WHERE id in (\
                            SELECT id from RECORDS order by id asc\
                             limit " + str(drop_len) + ")")
        tab_len = pa.record_limit - drop_len

    return tab_len


def train_policy(cursor,train_size):
    """ 训练集的选择 """
    # train_size = train_params['train_coff']*pa.min_trainsize
    # 训练次数不能超过 tab_len
    cursor.execute("SELECT num,x2 ,x4 ,temp,y1,y2  FROM RECORDS order by id desc limit "\
                        + str(train_size))

    # 先训练旧的，因为优先去除其中
    return cursor





class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    # def __init__(self, request, client_address, server):
    #     print("----------------------ThreadedTCPRequestHandler init.... ...")
    #     #socketserver.BaseRequestHandler.__init__(self, request, client_address, server)
    #     self.ensemble = Ensemble()
    #     self.ensemble.load_model()
    #     print("ThreadedTCPRequestHandler init ...")
    def setup(self): # 每次连接都会调用
        # self.ensemble = Ensemble()
        # self.ensemble.load_model()
        self.conn = sqlite3.connect(db_path)
        print("ThreadedTCPRequestHandler init ...")

    def handle(self):

        print("handeling...")
        data = self.request.recv(102400)
        jdata = json.loads(data,encoding="utf-8")     #编码转换
        # print( "test predict ",ensemble.predict(np.array([[1,1,1,1]])))
        # Response = {}
        cursor = self.conn.cursor()
        # TODO: 根据jdata params 参数确定运算类型
        if jdata['params'] =='predict':
            predicts = {}
            datasets = jdata['data']
            for id in datasets.keys():
                # print(id, jdata[id]["Num"],jdata[id]["x2"],jdata[id]["x4"],jdata[id]["temp"])
                featrues = np.array([[datasets[id]["Num"],datasets[id]["x2"],datasets[id]["x4"],datasets[id]["temp"]]])
                if jdata['method'] == 'ensemble':
                    predicts[id] = ensemble.predict(featrues)
                elif jdata['method'] == 'ann':
                    predicts[id] = ann.predict(featrues)
            # 下面是返回给client的json格式数据
            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'record':
            # cursor = self.conn.cursor()
            datasets = jdata['data']
            for id in datasets.keys():
                # print(id, jdata[id]["Num"],jdata[id]["x2"],jdata[id]["x4"],jdata[id]["temp"])
                featrues = \
                    [datasets[id]["Num"], datasets[id]["x2"], datasets[id]["x4"], \
                     datasets[id]["temp"],datasets[id]["y1"],datasets[id]["y2"]]
                cursor.execute("INSERT INTO records (num,x2,x4,temp,y1,y2) VALUES "
                          "(?,?,?,?,?,?)", \
                          (featrues[0], featrues[1], \
                           featrues[2], featrues[3], \
                           featrues[4], featrues[5]))
                # predicts[id] = ensemble.predict(featrues)
            # 下面是返回给client的json格式数据
            drop_len = int(pa.record_limit* pa.drop_policy )
            tab_len = deleteOldRecord(cursor, drop_len)
            # jresp = json.dumps(predicts)
            # tab_len = sqlQuery.execute("SELECT COUNT(ID) FROM RECORDS")
            self.conn.commit()
            cursor.execute("select max(id) from records")
            max_id = cursor.fetchone()

            predicts = {'record':True,\
                'insert_len':len(datasets),\
                        'record_len':tab_len, 'max_id': max_id, \
                        'record_limit':pa.record_limit } #'record finished!'

            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'clear':
            #truncate table set id eq zero
            # 清空数据库，设置 id位 0
            # cursor = self.conn.cursor()
            cursor.execute("DELETE FROM RECORDS")
            cursor.execute("UPDATE sqlite_sequence SET seq = 0 WHERE name = 'RECORDS' ")
            self.conn.commit()
            predicts = {'clear':True}
            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'optimize':
            # optimize model to find solve
            Num = jdata['Num']
            predicts ={'MachineNum':Num}
            if jdata['method'] == 'ann':
                x, y = solve(ann, Num)
                predicts.update({'solve': 'ann', \
                                 'x': x, 'y1': y[0], 'y2': y[1]})
            elif jdata['method']== 'ensemble':
                x,y =solve(ensemble,Num)
                predicts.update({'solve':'ensemble',\
                         'x':x,'y1':y[0],'y2':y[1]})

            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'train':
            # cursor = self.conn.cursor()
            tab_len = cursor.execute("SELECT COUNT(ID) FROM RECORDS") # 数据库记录数

            train_size = jdata['train_coff'] * pa.min_trainsize
            cursor = train_policy(cursor, train_size)
            if jdata['method'] == 'ensemble':
                ensemble.update_model(cursor,pa.batch_size)
            elif jdata['method'] == 'ann':
                ensemble.update_model(cursor,pa.batch_size)
            predicts = {'train': True, 'table_len': tab_len.fetchone(), \
                        'train_size':train_size,'min_trainsize':pa.min_trainsize,\
                        'batch_size':pa.batch_size,'max_trainsize': pa.max_trainsize }

            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))


    def __del__(self):
        self.conn.close()
        print("del fun run finished...")



 

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):

    pass

 

if __name__ == "__main__":

    # Port 0 means to select an arbitrary unused port
    HOST, PORT = "localhost", 10001

    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    ip, port = server.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target = server.serve_forever)

    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)
    print(" .... waiting for connection")

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
