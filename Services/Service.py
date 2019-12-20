import socket
import sys
import threading
import json
import numpy as np

from tag import train2

# nn=network.getNetWork()
# cnn = conv.main(False)
# 深度学习训练的神经网络,使用TensorFlow训练的神经网络模型，保存在文件中
nnservice = train2.NNService(model='model/20180731.ckpt-1000')

def main():
    # 创建服务器套接字
    serversocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    # 获取本地主机名称
    host = socket.gethostname()
    # 设置一个端口
    port = 12345
    # 将套接字与本地主机和端口绑定
    serversocket.bind((host,port))
    # 设置监听最大连接数
    serversocket.listen(5)
    # 获取本地服务器的连接信息
    myaddr = serversocket.getsockname()
    print("服务器地址:%s"%str(myaddr))
    # 循环等待接受客户端信息
    while True:
        # 获取一个客户端连接
        clientsocket,addr = serversocket.accept()
        print("连接地址:%s" % str(addr))
        try:
            t = ServerThreading(clientsocket)#为每一个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            pass
        pass
    serversocket.close()
    pass


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self,clientsocket,recvsize=1024*1024,encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")
        try:
            #接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg=msg[:-4]
                    break
            # 解析json格式的数据
            re = json.loads(msg)
            # 调用神经网络模型处理请求
            res = nnservice.hand(re['content'])
            sendmsg = json.dumps(res)
            # 发送数据
            self._socket.send(("%s"%sendmsg).encode(self._encoding))
            pass

        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close() 
        print("任务结束.....")
        pass

    def __del__(self):
           pass



if __name__ == "__main__":
    main()
