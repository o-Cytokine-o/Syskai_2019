from django.test import TestCase
import numpy as np
import assist_text  as assist

# Create your tests here
hand_arr = np.array([[9],[4,6,8],[1,10],[6,10,7],[8,8]])

print(assist.get_assist(hand_arr))