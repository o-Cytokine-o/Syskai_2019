# sakaguti関数とは、

def sakaguti(state_frag):

    text=[]
    d="ディーラーが全プレイヤーにカード配布するまでお待ちください。"
    p="それでは深い闇のゲームを開始します。"
    p1="プレイヤー1はアシスト画面に従い自身の行動を選択し宣言してください。"
    p2="プレイヤー1は自身の行動を終える場合、ターンエンドを宣言してください。"
    p3="プレイヤー2はアシスト画面に従い自身の行動を選択し宣言してください。"
    p4="プレイヤー2は自身の行動を終える場合、ターンエンドを宣言してください。"
    p5="プレイヤー3はアシスト画面に従い自身の行動を選択し宣言してください。"
    p6="プレイヤー3は自身の行動を終える場合、ターンエンドを宣言してください。"
    p7="プレイヤー4はアシスト画面に従い自身の行動を選択し宣言してください。"
    p8="プレイヤー4は自身の行動を終える場合、ターンエンドを宣言してください。"
    p9="ディーラーのカードの合計が、17以上になるまでカードを引き、その合計値で勝敗が決まります。"

    if state_frag==0:
        text.append(d)
    elif state_frag==1:
        text.append(p)
        text.append(p1)
        text.append(p2)    
    elif state_frag==2:
        text.append(p3)
        text.append(p4)
    elif state_frag==3:
        text.append(p5)
        text.append(p6)
    elif state_frag==4:
        text.append(p7)
        text.append(p8)
    elif state_frag==5:
        text.append(p9)
    elif state_frag==6:
        text.append()
    else:
        text.append("unko")

    return text

