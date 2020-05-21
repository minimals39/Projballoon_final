import socket
from threading import Thread
import time
class comm:
    def __init__(self,netip,port):
        #initialize the connection
        self.netip = netip
        self.port = port
        self.stopped = False
        self.decdata = None
        self.inp = 's'
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.initThread()

    def start(self):
        Thread(target=self.update,args = ()).start()
        #Thread(target=self.sendcommand,args = ()).start()
        return self

    def update(self):
        starttime=time.time()
        if self.stopped:
            return
        while True and self.stopped == False:
            try:
                self.getxyhf()
            except Exception as e:
                print("rec" + str(e))

    def getxyhf(self):
        try:
            self.data = self.s.recv(1024)
            self.decdata = self.data.decode()
        except:
            self.decdata = self.decdata


    def getxyh(self):
        if self.decdata:
            return self.decdata
        else: 
            return "0_0_0_0"

    def initThread(self):
        self.s.connect((self.netip, self.port))
        self.s.setblocking(0)
        #self.s.settimeout(10)
        self.stopped = False


    def setinp(self,inp):
        self.inp = inp

    def sendcommand(self):
        s = self.s
        try:
            print("sended")
            s.sendall(self.inp.encode())
        except Exception as e: 
            time.sleep(1)
            print("send" + str(e))


    t_send = lambda:Thread(target=sendcommand).start()

    def stop(self):
        self.stopped = True
        self.s.close()

