import urllib.parse

# sakaguti関数とは、

def sakaguti(field_state):

    text=[]
    d=urllib.parse.quote("ディーラーが全プレイヤーにカード配布するまでお待ちください。")
    p=urllib.parse.quote("それでは深い闇のゲームを開始します。")
    p1=urllib.parse.quote("プレイヤー1はアシスト画面に従い自身の行動を選択し宣言してください。")
    p2=urllib.parse.quote("プレイヤー1は自身の行動を終える場合、ターンエンドを宣言してください。")
    p3=urllib.parse.quote("プレイヤー2はアシスト画面に従い自身の行動を選択し宣言してください。")
    p4=urllib.parse.quote("プレイヤー2は自身の行動を終える場合、ターンエンドを宣言してください。")
    p5=urllib.parse.quote("プレイヤー3はアシスト画面に従い自身の行動を選択し宣言してください。")
    p6=urllib.parse.quote("プレイヤー3は自身の行動を終える場合、ターンエンドを宣言してください。")
    p7=urllib.parse.quote("プレイヤー4はアシスト画面に従い自身の行動を選択し宣言してください。")
    p8=urllib.parse.quote("プレイヤー4は自身の行動を終える場合、ターンエンドを宣言してください。")
    p9=urllib.parse.quote("ディーラーのカードの合計が、17以上になるまでカードを引き、その合計値で勝敗が決まります。")

    if 0 in field_state:
        text.append(d)
    elif field_state[1]==1:
        text.append(p)
        text.append(p1)
        text.append(p2)    
    elif field_state[2]==1:
        text.append(p3)
        text.append(p4)
    elif field_state[3]==1:
        text.append(p5)
        text.append(p6)
    elif field_state[4]==1:
        text.append(p7)
        text.append(p8)
    elif field_state[0]==3:
        text.append(p9)
    elif field_state[-1]==3:
        text.append()
    else:
        text.append("unko")

    return text