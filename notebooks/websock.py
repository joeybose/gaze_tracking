import tornado.httpserver
import tornado.websocket
import tornado.ioloop
import tornado.web
import socket
from multiprocessing import Process, Pipe
import datetime

POLLING_INTERVAL = 10

wss = []
class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print 'Online'
        if self not in wss:
            wss.append(self)

    def on_close(self):
        print 'Offline'
        if self in wss:
            wss.remove(self)
      
    def on_message(self, message):
        print 'message received:  %s' % message
        # Reverse Message and send it back
        print 'sending back message: %s' % message[::-1]
        self.write_message(message[::-1])
 
    def check_origin(self, origin):
        return True

application = tornado.web.Application([
    (r'/ws', WSHandler),
])
    
def wsSend():
    if conn.poll :
        for ws in wss:
            if not ws.ws_connection.stream.socket:
                print "Web socket is not live!!!"
            else:
                ws.write_message(conn.recv())
            
def setup(child_conn):
    global conn 
    conn = child_conn
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(2000)
    myIP = socket.gethostbyname(socket.gethostname())
    print '*** Websocket Server Started at %s***' % myIP
    main_loop = tornado.ioloop.IOLoop.instance()
    tornado_start= tornado.ioloop.PeriodicCallback(wsSend,POLLING_INTERVAL, io_loop=main_loop)
    tornado_start.start()
    main_loop.start()
