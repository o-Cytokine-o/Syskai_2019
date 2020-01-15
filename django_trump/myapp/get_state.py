def get_state(field_list,turn,field_state):
    #field_state[ディーラー、プレイヤー１、プレイヤー２、プレイヤー３、プレイヤー４]
    #0:カードがまだ2枚配られていない
    #1:カードが2枚ある
    #2:そのプレイヤーのターン終了した
    #3:全プレイヤーのターンが終了した
    
    cnt=0 #カウンタ初期化
    
    #カード配っていない
    if 3 not in field_state: #ゲーム終了フラグでないとき
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
            if (len(i)==2)and(field_state[cnt]==0):
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
                    turn=False
                    break
            cnt+=1

    if field_state.count(2)==5:
       field_state[0]=3

    if field_state[0]==3 and sum(field_list[0])>=17:
        field_state = [3 for i in range(5)]
    
    return field_state,turn