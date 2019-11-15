from django.shortcuts import render

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

