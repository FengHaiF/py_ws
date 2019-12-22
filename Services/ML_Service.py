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
from Regresser import Ensemble
from parameters import Parameters

warnings.filterwarnings(action='ignore', \
             module='sklearn')

pa = Parameters()

# 导入ensemble 的模型
ensemble = Ensemble()
ensemble.load_model()

db_path = 'db/test_records.db'

def deleteOldRecord(cursor,drop_len):
    tab_len = cursor.execute("SELECT count(id) from RECORDS")
    tab_len = list(tab_len)[0][0]
    if tab_len > pa.record_limit:
        cursor.execute("DELETE FROM RECORDS WHERE id in (\
                            SELECT id from RECORDS order by id asc\
                             limit " + str(drop_len) + ")")
        tab_len = pa.record_limit - drop_len

    return tab_len






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
        # TODO: 根据jdata params 参数确定运算类型
        if jdata['params'] =='predict':
            predicts = {}
            datasets = jdata['data']
            for id in datasets.keys():
                # print(id, jdata[id]["Num"],jdata[id]["x2"],jdata[id]["x4"],jdata[id]["temp"])
                featrues = np.array([[datasets[id]["Num"],datasets[id]["x2"],datasets[id]["x4"],datasets[id]["temp"]]])
                predicts[id] = ensemble.predict(featrues)
            # 下面是返回给client的json格式数据
            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'record':
            cursor = self.conn.cursor()
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
            drop_len = int( pa.record_limit* pa.drop_policy )
            tab_len = deleteOldRecord(cursor, drop_len)
            # jresp = json.dumps(predicts)
            # tab_len = sqlQuery.execute("SELECT COUNT(ID) FROM RECORDS")
            self.conn.commit()
            predicts = {'flag':True,\
                'insert_len':len(datasets),\
                        'record_len':tab_len,\
                        'record_limit':pa.record_limit } #'record finished!'

            jresp = json.dumps(predicts)
            print(jresp)
            self.request.sendall(jresp.encode('utf-8'))

        elif jdata['params'] == 'train':
            sqlQuery = self.conn.cursor()
            tab_len = sqlQuery.execute("SELECT COUNT(ID) FROM RECORDS") # 数据库记录数
            predicts = {'flag':True,'table_len':tab_len,\
                        'tab_limit':20000}
            ensemble.update_model(sqlQuery,pa.batch_size)

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
