import urllib.parse

# sakaguti関数とは、

def sakaguti(field_state):

    text=[]
    d=urllib.parse.quote("ディーラーが全プレイヤーにカード配布するまでお待ちください。")
    p=urllib.parse.quote("それでは深い闇のゲームを開始します。")
    p1=urllib.parse.quote("プレイヤー1はヒット、スタンド、サレンダーのいずれかを選択して下さい")
    p1_5=urllib.parse.quote("サレンダーをする場合「サレンダー」と宣言してください")
    p2=urllib.parse.quote("ヒット、スタンドをする場合、ディーラーに伝え、「ターンエンド」を宣言してください。")
    p3=urllib.parse.quote("プレイヤー2はヒット、スタンド、サレンダーのいずれかを選択して下さい")
    p4=urllib.parse.quote("ヒット、スタンドをする場合、ディーラーに伝え、「ターンエンド」を宣言してください。")
    p5=urllib.parse.quote("プレイヤー3はヒット、スタンド、サレンダーのいずれかを選択して下さい")
    p6=urllib.parse.quote("ヒット、スタンドをする場合、ディーラーに伝え、「ターンエンド」を宣言してください。")
    p7=urllib.parse.quote("プレイヤー4はヒット、スタンド、サレンダーのいずれかを選択して下さい")
    p8=urllib.parse.quote("ヒット、スタンドをする場合、ディーラーに伝え、「ターンエンド」を宣言してください。")
    p9=urllib.parse.quote("ディーラーのカードの合計が、17以上になるまでカードを引き、その合計値で勝敗が決まります。")
    p10=urllib.parse.quote("ゲームが終了しました。結果は各プレイヤーの手札の上に表示されています。")

    p2_5 = urllib.parse.quote("どうすればいいか分らないときはアシスト画面も参考にしてください！")

    if 0 in field_state:
        text.append(d)
    elif field_state[1]==1:
        text.append(p)
        text.append(p1)
        text.append(p1_5)
        text.append(p2)
        text.append(p2_5)
    elif field_state[2]==1:
        text.append(p3)
        text.append(p1_5)
        text.append(p4)
        text.append(p2_5)
    elif field_state[3]==1:
        text.append(p5)
        text.append(p1_5)
        text.append(p6)
        text.append(p2_5)
    elif field_state[4]==1:
        text.append(p7)
        text.append(p1_5)
        text.append(p8)
        text.append(p2_5)
    elif field_state[0]==3:
        text.append(p9)
    elif field_state[0] == 3 and field_state[-1] == 3:
        text.append(P10)
    else:
        text.append("unko")

    return text