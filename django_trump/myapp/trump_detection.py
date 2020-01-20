import os
import tensorflow as tf
import numpy as np
import cv2
import time
from myapp import tokui
from myapp import get_state
from myapp import assist
from myapp import client as cli
from myapp import speechtext as sptxt
import threading
from multiprocessing import Process
from multiprocessing import Value
from myapp.object_detection.models7.research.object_detection.utils import label_map_util
from myapp.object_detection.models7.research.object_detection.utils import visualization_utils as vis_util
from websocket import create_connection
from django.views.generic import TemplateView

#ターンエンドフラグの初期化
turn_frag = Value('b',False)
surrender_frag = Value('b',False)
surrender_list = [0,0,0,0]

#サレンダーを宣言した場合のフラグ処理
def surrender_def():
    if surrender_frag.value:
        cnt = 0
        for i in field_state:
                if cnt>0:
                    pre_player=field_state[cnt-1]
                    if (i==1)and(pre_player==2):
                        field_state[cnt]=2
                        surrender_list[cnt] = 1
                        surrender_frag.value = False
                        break
                cnt+=1

#multiprocessで実行させるためのラッパ関数
def sptxtDef(turn_frag):
    while True:
        f = sptxt.SpeechToText()
        if f == 1:
            print("mmmmmmmmmmmmmmmmmmmグローバル変更成功mmmmmmmmmmmmmmmm")
            turn_frag.value = True

        elif f == 2:
            print("mmmmmmmmmmmmmmmmmmmサレンダー変更成功mmmmmmmmmmmmmmmm")
            turn_frag.value = True
            surrender_frag.value = True



#テキストから数値を返す関数 'ACE'→1,'two'→2
def trump_text_to_num(str1):
    class_names = ['ace','two','three','four','five','six','seven','eight','nine','ten','jack','queen','king']
    if int(class_names.index(str1)+1) >= 10:
        return 10
    else:
        return int(class_names.index(str1)+1)


def gen(camera,tut_frag):

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,'myapp','object_detection','models7','research','object_detection','inference_graph','frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'myapp','object_detection','models7','research','object_detection','training','labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 13

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    #データ送信用クライアントの処理
    ws = create_connection("ws://localhost:9990/")

    #ゲームのステータスを初期化
    field_state = [0,0,0,0,0]

    #勝敗結果表示用フォントパラメータ
    font_color = (0, 181, 48)
    back_color = (255,255,255)
    back_th = 15
    front_th = 5
    font_size = 2

    #ターンエンド宣言フラグ`取得関数のの並列実行
    turn_end_thread = Process(target=sptxtDef, args=[turn_frag,])
    turn_end_thread.start()
    
    while(True):

        ret, frame = camera.video.read()
        height = frame.shape[0]
        width = frame.shape[1]
        print(height,width)
        frame = frame[int(height*0.13):int(height*0.88),int(width*0.15):int(width*0.85)]
        frame_expanded = np.expand_dims(frame, axis=0)


        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        frame,box,box_to_color_map = vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        #ここに認識したカードの座標とカードの種類を持ってくる
        #その情報からプレイヤーのカードの合計かを処理する

        #frameからプレイヤーの座標を取得
        #そこにスコア（合計値）を表示する


        #各プレイヤーの数値を初期化
        num_dea = []
        num_p1 = []
        num_p2 = []
        num_p3 = []
        num_p4 = []

        #合計値の初期化
        total_num_dea = 0
        total_num_p1 = 0
        total_num_p2 = 0
        total_num_p3 = 0
        total_num_p4 = 0

        total_num_deas = 0
        total_num_p1s = 0
        total_num_p2s = 0
        total_num_p3s = 0
        total_num_p4s = 0

        #認識したカードの数字を合計する
        for item in box.items():
            text = item[1][0].split(':')

            ymin, xmin, ymax, xmax = item[0]

            #プレイヤーかディーラーかの判定
            if (ymax - ymin / 2)>0.5:
                #どのプレイヤーのカードなのかを判定
                if xmax<0.25:
                    player_n =  'プレイヤー１'
                    num_p1.append(trump_text_to_num(text[0]))
                    total_num_p1 = total_num_p1 + (trump_text_to_num(text[0]))
                    #手札にAが入っていた場合の考慮
                    if 1 in num_p1:
                        total_num_p1s = (total_num_p1 - 1) + 11
                    
                elif xmax<0.5:
                    player_n =  'プレイヤー2'
                    num_p2.append(trump_text_to_num(text[0]))
                    total_num_p2 = total_num_p2 + (trump_text_to_num(text[0]))
                    #手札にAが入っていた場合の考慮
                    if 1 in num_p2:
                        total_num_p2s = (total_num_p2 - 1) + 11
                    
                elif xmax<0.75:
                    player_n =  'プレイヤー3'
                    num_p3.append(trump_text_to_num(text[0]))
                    total_num_p3 = total_num_p3 + (trump_text_to_num(text[0]))
                    #手札にAが入っていた場合の考慮
                    if 1 in num_p3:
                        total_num_p3s = (total_num_p3 - 1) + 11
                    
                else:
                    player_n =  'プレイヤー4'
                    num_p4.append(trump_text_to_num(text[0]))
                    total_num_p4 = total_num_p4 + (trump_text_to_num(text[0]))
                    #手札にAが入っていた場合の考慮
                    if 1 in num_p4:
                        total_num_p4s = (total_num_p4 - 1) + 11

            else:
                num_dea.append(trump_text_to_num(text[0]))
                total_num_dea = total_num_dea + (trump_text_to_num(text[0]))
                total_num_deas = total_num_deas + (trump_text_to_num(text[0])) + 10

        #全プレイヤーのカードのデータ
        if len(num_dea)==0:
            num_dea = [0]
        if len(num_p1)==0:
            num_p1 = [0]
        if len(num_p2)==0:
            num_p2 = [0]
        if len(num_p3)==0:
            num_p3 = [0]
        if len(num_p4)==0:
            num_p4 = [0]

        field_list = np.array([num_dea,num_p1,num_p2,num_p3,num_p4])
        print('デバッグ：field_list'+str(field_list))

        height = frame.shape[0]
        width = frame.shape[1]

        #ゲームが終わったときに勝敗を表示する
        if field_state[0] == 3 and field_state[-1] == 3:

            #プレイヤー１の判定
            if 1 in field_list[1]:
                if 1 in field_list[0]:
                    if (total_num_p1s <= 21) and (total_num_p1s > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1s <= 21) and (total_num_p1s == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1s > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p1s <= 21) and (total_num_p1s > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1s <= 21) and (total_num_p1s == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1s > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
            
            else:
                if 1 in field_list[0]:
                    if (total_num_p1 <= 21) and (total_num_p1 > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1 <= 21) and (total_num_p1 == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1 > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p1 <= 21) and (total_num_p1 > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1 <= 21) and (total_num_p1 == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p1 > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)

            #プレイヤー２の判定
            if 1 in field_list[2]:
                if 1 in field_list[0]:
                    if (total_num_p2s <= 21) and (total_num_p2s > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2s <= 21) and (total_num_p2s == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2s > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p2s <= 21) and (total_num_p2s > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2s <= 21) and (total_num_p2s == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2s > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
            
            else:
                if 1 in field_list[0]:
                    if (total_num_p2 <= 21) and (total_num_p2 > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2 <= 21) and (total_num_p2 == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2 > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p2 <= 21) and (total_num_p2 > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2 <= 21) and (total_num_p2 == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p2 > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)

            #プレイヤー3の判定
            if 1 in field_list[3]:
                if 1 in field_list[0]:
                    if (total_num_p3s <= 21) and (total_num_p3s > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3s <= 21) and (total_num_p3s == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3s > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p3s <= 21) and (total_num_p3s > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3s <= 21) and (total_num_p3s == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3s > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
            
            else:
                if 1 in field_list[0]:
                    if (total_num_p3 <= 21) and (total_num_p3 > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3 <= 21) and (total_num_p3 == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3 > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p3 <= 21) and (total_num_p3 > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3 <= 21) and (total_num_p3 == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p3 > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)

            #プレイヤー4の判定
            if 1 in field_list[4]:
                if 1 in field_list[0]:
                    if (total_num_p4s <= 21) and (total_num_p4s > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4s <= 21) and (total_num_p4s == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4s > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p4s <= 21) and (total_num_p4s > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4s <= 21) and (total_num_p4s == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4s > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
            
            else:
                if 1 in field_list[0]:
                    if (total_num_p4 <= 21) and (total_num_p4 > total_num_deas) or (21 < total_num_deas):
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4 <= 21) and (total_num_p4 == total_num_deas):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4 > 21) and (total_num_deas > 21):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                
                else:
                    if (total_num_p4 <= 21) and (total_num_p4 > total_num_dea) or (21 < total_num_dea):
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'win', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4 <= 21) and (total_num_p4 == total_num_dea):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    elif (total_num_p4 > 21) and (total_num_dea > 21):
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'drow', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
                    else:
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)
                        cv2.putText(frame, 'lose', (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, front_th)
        
        #映像にそれぞれのプレイヤーの手札の合計値を表示する
        else:

            cv2.putText(frame, str(total_num_p1), (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p1), (int(width*0.125), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 5)
            cv2.putText(frame, str(total_num_p2), (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p2), (int(width*0.375), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 5)
            cv2.putText(frame, str(total_num_p3), (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p3), (int(width*0.625), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 5)
            cv2.putText(frame, str(total_num_p4), (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p4), (int(width*0.875), int(height*0.75)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 5)

            #手札にAがあった場合
            cv2.putText(frame, str(total_num_p1s), (int(width*0.125), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p1s), (int(width*0.125), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), 5)
            cv2.putText(frame, str(total_num_p2s), (int(width*0.375), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p2s), (int(width*0.375), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), 5)
            cv2.putText(frame, str(total_num_p3s), (int(width*0.625), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p3s), (int(width*0.625), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), 5)
            cv2.putText(frame, str(total_num_p4s), (int(width*0.875), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, back_color, back_th)#輪郭用
            cv2.putText(frame, str(total_num_p4s), (int(width*0.875), int(height*0.85)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 255), 5)

        #ゲームの状態のフラグを取得
        #チュートリアルありとなしの各フラグ処理
        if tut_frag:
            field_state,turn_frag.value = get_state.get_state(field_list,bool(turn_frag.value),field_state)
        else:
            field_state,turn_frag.value = get_state.get_state(field_list,True,field_state)
        print('デバッグ：field_state'+str(field_state))

        #サレンダーのフラグ処理
        surrender_def()
        #取得したフラグからチュートリアルのテキストを取得
        tutorial_text = tokui.sakaguti(field_state)
        
        print("マイクデバッグ："+str(bool(turn_frag.value)))

        #全プレイヤーの手札をもとに戦術の結果を取得する
        #カードが配布された後、アシストを表示する
        if field_state[0]>=2:
            assist_text = assist.get_assist(field_list,surrender_list)
        else:
            assist_text = ''

        #チュートリアルとアシストの結果をAjaxのサーバに送信する
        cli.send_data(ws,tutorial_text,assist_text)

        ret,jpeg = cv2.imencode('.jpg',frame)
        frame = jpeg.tobytes()
        
        yield(b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        time.sleep(0.1)
    
    ws.close()