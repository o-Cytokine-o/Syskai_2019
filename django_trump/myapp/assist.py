from myapp import logic as lg
import numpy as np
import urllib.parse

def get_assist(hand_arr,surrender_list):

    assist_text = []

    d_list = hand_arr[0]
    p_list = hand_arr[1:]
    i = 1
    for player_hand in p_list:
        
        assist_flag = 0 

        assist_flag = lg.assist(player_hand,d_list)

        if surrender_list[i-1] == 1:
            assist_text.append(urllib.parse.quote("プレイヤー"+str(i)+"：\tサレンダー済み"))

        else:
            if assist_flag == 0:
                assist_text.append(urllib.parse.quote("プレイヤー"+str(i)+"：\tスタンド"))
            
            elif assist_flag == 1:
                assist_text.append(urllib.parse.quote("プレイヤー"+str(i)+"：\tヒット"))

            elif assist_flag == 2:
                assist_text.append(urllib.parse.quote("プレイヤー"+str(i)+"：\tサレンダー"))
            
            else:
                assist_text.append(urllib.parse.quote("プレイヤー"+str(i)+"：\tバースト"))

        i = i + 1

    return assist_text

        