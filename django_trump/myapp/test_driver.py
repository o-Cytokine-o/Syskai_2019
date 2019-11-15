import test_data_stub as td1
from websocket import create_connection
import client as cli

ws = create_connection("ws://localhost:9990/")

for i in td1.return_data():
    print('ドライバ'+str(i))
    cli.send_data(ws,i)

ws.close()