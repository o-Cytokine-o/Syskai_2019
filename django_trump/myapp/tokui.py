# sakaguti関数とは、

def sakaguti(field_state):

    text=[]
    d="ディーラーが全プレイヤーにカード配布するまでお待ちください。aqwerty"
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

    if field_state[0]==0:
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

def get_state(field_list,turn,field_state):
    #field_state[ディーラー、プレイヤー１、プレイヤー２、プレイヤー３、プレイヤー４]
    #0:カードがまだ2枚配られていない
    #1:カードが2枚ある
    #2:そのプレイヤーのターン終了した
    #3:全プレイヤーのターンが終了した
    
    
    
    """ global field_state #関数内グローバル変数宣言
    global turn_end  """
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
            if (len(i)==1)and(field_state[cnt]==0)and(field_list[0][0]!=0):
                field_state[cnt]=2
        else:    
            if (len(i)==2)and(field_state[cnt]!=2):
                field_state[cnt]=1
        cnt+=1
    
    #ターンエンド判定
    cnt=0
    if turn==True:
        for i in field_state:
            if cnt>0:
                pre_player=field_state[cnt-1]
                if (i==1)and(pre_player==2):
                    field_state[cnt]=2
                    turn_end=False
                    break
            cnt+=1

    if field_state.count(2)==5:
       field_state[0]=3

    if field_state[0]==3 and field_list[0]>=17:
        field_state = [3 for i in range(5)]
    
    return field_state