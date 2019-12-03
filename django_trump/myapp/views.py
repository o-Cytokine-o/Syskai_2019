from myapp import trump_detection as trd
from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse,HttpResponseServerError
import cv2
from django.views.decorators import gzip

# Create your views here.

def index(request):
    params = { # <- 渡したい変数を辞書型オブジェクトに格納
        'title': 'Hi Django!',
        'subtitle': 'This is my 1st Django app.',
    }
    return render(request, 'index.html', params) # <- 引数にparamsを追記


def game(request):
    params = { # <- 渡したい変数を辞書型オブジェクトに格納
        'title': 'Hi Django!',
        'subtitle': 'This is my 1st Django app.',
    }
    return render(request,'game.html', params)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        ret = self.video.set(3,900)
        ret = self.video.set(4,677)
        
    def __del__(self):
        self.video.release()

@gzip.gzip_page
def view_OD(request): 
    try:
        return StreamingHttpResponse(trd.gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")
    except HttpResponseServerError as e:
        print("aborted")