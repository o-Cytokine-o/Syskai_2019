def sakaguti():


    frag = 0
    hoge1 = 1
    hoge2 = 0
    hoge3 = 0
    hoge4 = 0
    hoge5 = 0
    text=[]
    d="ディーラーは各プレイヤーにカード配布してください。"
    p1="プレイヤー1はアシスト画面に従い自身の行動を選択してください。"
    p2="プレイヤー1は終了する場合は終了を宣言してください。"
    p3="プレイヤー2はアシスト画面に従い自身の行動を選択してください。"
    p4="プレイヤー2は終了する場合は終了を宣言してください。"
    p5="プレイヤー3はアシスト画面に従い自身の行動を選択してください。"
    p6="プレイヤー3は終了する場合は終了を宣言してください。"
    p7="プレイヤー4はアシスト画面に従い自身の行動を選択してください。"
    p8="プレイヤー4は終了する場合は終了を宣言してください。"

    if frag==hoge1:
        text.append(d)
    elif frag==hoge2:
        text.append(p1)
        text.append(p2)
    elif frag==hoge3:
        text.append(p3)
        text.append(p4)
    elif frag==hoge4:
        text.append(p5)
        text.append(p6)
    elif frag==hoge5:
        text.append(p7)
        text.append(p8)
    else:
        text.append("unko")


    return text

def get_state(field_list,turn):
    global field_state #関数内グローバル変数宣言
    global turn_end #エラー吐くお
    cnt=0 #カウンタ初期化
    
    #カード配っていない
    for i in field_list:
        if (0 in i)or(len(i)==1):
            field_state[cnt]=0
        cnt+=1
    
    #全員配り終わった
    cnt=0
    for i in field_list:
        if cnt==0:
            if (len(i)==1)and(field_state[cnt]==0):
                field_state[cnt]=1
        else:    
            if (len(i)==2)and(field_state[cnt]!=2):
                field_state[cnt]=1
        cnt+=1
    
    #ターンエンド判定
    cnt=0
    if turn==True:
        for i in field_state:
            if cnt>0:
                if i==1:
                    field_state[cnt]=2
                    turn_end=False
                    break
            cnt+=1

    if field_state.count(2)==4:
       field_state[cnt]=3
    
    return field_state
