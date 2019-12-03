from myapp import logic as lg
import numpy as np
import urllib.parse

def get_assist(hand_arr):

    assist_text = []

    d_list = hand_arr[0]
    p_list = hand_arr[1:]
    i = 1
    for player_hand in p_list:
        
        assist_flag = 0 

        assist_flag = lg.assist(player_hand,d_list)

        if assist_flag == 0:
            assist_text.append(urllib.parse.quote("プレイヤー"+i+"：スタンド"))
        
        elif assist_flag == 1:
            assist_text.append(urllib.parse.quote("プレイヤー"+i+"ヒット"))

        elif assist_flag == 2:
            assist_text.append(urllib.parse.quote("プレイヤー"+i+"サレンダー"))
        
        else:
            assist_text.append(urllib.parse.quote("プレイヤー"+i+"バースト"))

        i = i + 1

    return assist_text

        