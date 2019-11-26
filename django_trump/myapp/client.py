import time
from websocket import create_connection

def send_data(ws,data1,data2):
    print('クライアントデバッグ：'+str(data1)+':'+str(data2))
    ws.send(str(data1)+':'+str(data2))