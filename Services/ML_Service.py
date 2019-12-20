#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-
#
import socket
import threading
import socketserver
import json, types,string
import os, time
import numpy as np
import warnings
from Regresser import Ensemble

warnings.filterwarnings(action='ignore', \
             module='sklearn')

# 导入ensemble 的模型
ensemble = Ensemble()
ensemble.load_model()


class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    # def __init__(self, request, client_address, server):
    #     print("----------------------ThreadedTCPRequestHandler init.... ...")
    #     #socketserver.BaseRequestHandler.__init__(self, request, client_address, server)
    #     self.ensemble = Ensemble()
    #     self.ensemble.load_model()
    #     print("ThreadedTCPRequestHandler init ...")

    def handle(self):
        print("handeling...")
        data = self.request.recv(102400)
        jdata = json.loads(data,encoding="utf-8")     #编码转换
        # print( "test predict ",ensemble.predict(np.array([[1,1,1,1]])))
        predicts = {}
        for id in jdata.keys():
            # print(id, jdata[id]["Num"],jdata[id]["x2"],jdata[id]["x4"],jdata[id]["temp"])
            featrues = np.array([[jdata[id]["Num"],jdata[id]["x2"],jdata[id]["x4"],jdata[id]["temp"]]])
            predicts[id] = ensemble.predict(featrues)
        # 下面是返回给client的json格式数据
        jresp = json.dumps(predicts)
        print(jresp)
        self.request.sendall(jresp.encode('utf-8'))

 

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
