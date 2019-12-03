import time
import urllib.parse
from websocket import create_connection

def send_data(ws,data1,data2):
    print('クライアントデバッグ：'+ urllib.parse.unquote(str(data1))+':'+urllib.parse.unquote(str(data2)))
    #dataに入っている文字列を全て組み合わせる
    text1 = ''
    text2 = ''
    for st in data1:
        text1 = text1 + ',' + st
    text1 = text1[1:]
    for st in data2:
        text2 = text2 + ',' + st
    text2 = text2[1:]
    ws.send(str(text1)+':'+str(text2))