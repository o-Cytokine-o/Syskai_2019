from django.test import TestCase
import numpy as np

import tokui

# Create your tests here
def trump_text_to_num(str1):
    class_names = ['ace','two','three','four','five','six','seven','eight','nine','ten','jack','queen','king']
    if int(class_names.index(str1)+1) >= 10:
        return 10
    else:
        return int(class_names.index(str1)+1)
hand_arr = np.array([[9],[4,6],[1,10],[6,10],[8,8]])
field_state = [0,0,0,0,0]
#ターンエンドフラグの初期化
turn_end = False

#print(assist.get_assist(hand_arr))

#print(tokui.get_state(hand_arr,turn_end,field_state))

print(tokui.sakaguti(field_state))

print(trump_text_to_num('two'))