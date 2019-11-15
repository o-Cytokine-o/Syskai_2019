import numpy as np

def soft_hand(p_list,d_list):
    
    dcard=d_list
    pcard=p_list

    p_sum=sum(pcard)+10

    #print(p_sum)

    #ディーラー側リスト作成　ループで改良の余地あり
    list1=np.array([[1] for i in range(2)])
    list2=np.array([[2] for i in range(3)])
    list3=np.array([[3] for i in range(2)])
    list4=np.array([[4] for i in range(2)])
    list5=np.array([[5] for i in range(2)])
    list6=np.array([[6] for i in range(2)])
    list7=np.array([[7] for i in range(3)])
    list8=np.array([[8] for i in range(3)])
    list9=np.array([[9] for i in range(2)])
    list10=np.array([[10] for i in range(2)])

    #プレイヤー側リスト作成
    list_p1=np.array([[18],[19],[20]])
    list_p2=np.array([[19],[20]])

    #結合
    list1_p2=np.concatenate([list1, list_p2], 1)
    list2_p1=np.concatenate([list2, list_p1], 1)
    list3_p2=np.concatenate([list3, list_p2], 1)
    list4_p2=np.concatenate([list4, list_p2], 1)
    list5_p2=np.concatenate([list5, list_p2], 1)
    list6_p2=np.concatenate([list6, list_p2], 1)
    list7_p1=np.concatenate([list7, list_p1], 1)
    list8_p1=np.concatenate([list8, list_p1], 1)
    list9_p2=np.concatenate([list9, list_p2], 1)
    list10_p2=np.concatenate([list10, list_p2], 1)



    #ハードリスト作成　改良の余地あり

    a = np.r_['0', list1_p2, list2_p1]
    b = np.r_['0', list3_p2, list4_p2]
    c = np.r_['0', list5_p2, list6_p2]
    d = np.r_['0', list7_p1, list8_p1]
    e = np.r_['0', list9_p2, list10_p2]
    f = np.r_['0', a, b]
    g = np.r_['0', c, d]
    h = np.r_['0', f, g]
    soft_list = np.r_['0', h, e]

    #print(soft_list)

    #フラグリスト作成
    frag1=np.array([[0] for i in range(2)])
    frag2=np.array([[0] for i in range(3)])
    frag3=np.array([[0] for i in range(2)])
    frag4=np.array([[0] for i in range(2)])
    frag5=np.array([[0] for i in range(2)])
    frag6=np.array([[0] for i in range(2)])
    frag7=np.array([[0] for i in range(3)])
    frag8=np.array([[0] for i in range(3)])
    frag9=np.array([[0] for i in range(2)])
    frag10=np.array([[0] for i in range(2)])

    f1 = np.append(frag1, frag2)
    f2 = np.append(frag3, frag4)
    f3 = np.append(frag5, frag6)
    f4 = np.append(frag7, frag8)
    f5 = np.append(frag9, frag10)
    f6 = np.append(f1, f2)
    f7 = np.append(f3, f4)
    f8 = np.append(f6, f7)
    frag_list = np.append(f8, f5)

    #print(frag_list)


    #組み合わせ検索　戻り値＝フラグ
    #stand=0,sur=2
    retu=np.where((soft_list[:,0]==dcard)&(soft_list[:,1]==p_sum),True,False)

    if True in retu:
        result=np.where((soft_list[:,0]==dcard)&(soft_list[:,1]==p_sum))
        return  int(frag_list[result])
    else:
        result=1
        return result

def hard_hand(p_list,d_list):
    dcard=d_list
    pcard=p_list

    p_sum=sum(pcard)



    #ディーラー側リスト作成　ループで改良の余地あり
    list1=np.array([[1]])
    list2=np.array([[2] for i in range(4)])
    list3=np.array([[3] for i in range(4)])
    list4=np.array([[4] for i in range(5)])
    list5=np.array([[5] for i in range(5)])
    list6=np.array([[6] for i in range(5)])
    list9=np.array([[9]])
    list10=np.array([[10] for i in range(2)])

    #プレイヤー側リスト作成
    list_p1=np.array([[13],[14],[15],[16]])
    list_p2=np.array([[12],[13],[14],[15],[16]])
    list_p3=np.array([[16]])
    list_p4=np.array([[15],[16]])

    #結合
    list1_p3=np.concatenate([list1, list_p3], 1)
    list2_p1=np.concatenate([list2, list_p1], 1)
    list3_p1=np.concatenate([list3, list_p1], 1)
    list4_p2=np.concatenate([list4, list_p2], 1)
    list5_p2=np.concatenate([list5, list_p2], 1)
    list6_p1=np.concatenate([list6, list_p2], 1)
    list9_p3=np.concatenate([list9, list_p3], 1)
    list10_p4=np.concatenate([list10, list_p4], 1)

    #ハードリスト作成　改良の余地あり

    a = np.r_['0', list1_p3, list2_p1]
    b = np.r_['0', list3_p1, list4_p2]
    c = np.r_['0', list5_p2, list6_p1]
    d = np.r_['0', list9_p3, list10_p4]
    e = np.r_['0', a, b]
    f  = np.r_['0', c, d]
    hard_list = np.r_['0', e, f]

    #print(hard_list)

    #フラグリスト作成
    frag1=np.array([[2]])
    frag2=np.array([[0] for i in range(4)])
    frag3=np.array([[0] for i in range(4)])
    frag4=np.array([[0] for i in range(5)])
    frag5=np.array([[0] for i in range(5)])
    frag6=np.array([[0] for i in range(5)])
    frag9=np.array([[2]])
    frag10=np.array([[2] for i in range(2)])

    f1 = np.append(frag1, frag2)
    f2 = np.append(frag3, frag4)
    f3 = np.append(frag5, frag6)
    f4 = np.append(frag9, frag10)
    f5 = np.append(f1, f2)
    f6 = np.append(f3, f4)
    frag_list = np.append(f5, f6)

    #組み合わせ検索　戻り値＝フラグ
    retu=np.where((hard_list[:,0]==dcard)&(hard_list[:,1]==p_sum),True,False)

    if True in retu:
        result=np.where((hard_list[:,0]==dcard)&(hard_list[:,1]==p_sum))
        return  int(frag_list[result])
    else:
        result=1
        return result

def assist(p_list,d_list):
    #assit_frag:stand=0,hit=1,surrender=2
    
    #fragをhitに設定
    assist_frag=1

    pcard=p_list
    p_sum=sum(pcard)

    if p_sum>21: #バースト判定
        assist_frag=4
    elif 1 in p_list: #1がある
        assist_frag=soft_hand(p_list,d_list)
    elif p_sum>16: #17以上
        assist_frag=1
    else:
        assist_frag=hard_hand(p_list,d_list)

    return assist_frag
    

