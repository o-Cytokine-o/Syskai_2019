import time
from websocket import create_connection

def send_data(ws,data):
    print('クライアント'+str(data))
    ws.send(str(data))