#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# -*- coding:utf-8 -*-
#
import socket
import threading
import socketserver
import json, types,string
import os, time

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def handle(self):
        data = self.request.recv(102400)
        jdata = json.loads(data,encoding="utf-8")     #编码转换
        for date in jdata.keys():
            print(date.encode('utf-8'), jdata[date]["titles"][0],jdata[date]["titles"][1],jdata[date]["titles"][2])
        # 下面是返回给client的json格式数据
        topWords = {}
        topWords['南海'] = 0.123
        topWords['南沙'] = 0.543
        response = {}

        response['nextDayHot'] = 123
        response['topWords'] = topWords

        jresp = json.dumps(response)
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
