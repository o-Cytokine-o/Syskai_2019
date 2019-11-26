# 概要

webカメラを使ってトランプを認識したり手札を認識したりして、場面を把握しゲームをプレイする。
またはプレイヤーのサポートをする。

# Installation

## 1.Anacondaをインストールし環境を作る

### 1-1.Python[3.6.9]の環境を作る

### 1-2.必要なライブラリのインストール
tensorflowをインストール
[PCにグラボがある場合]
```
pip install tensorflow-gpu==1.14
```
[PCにグラボがない場合]
```
pip install tensorflow==1.14
```
各種ライブラリのインストール
```
pip install numpy
pip install lxml
pip install opencv-python
pip install matplotlib
pip install pillow
```

## 2.任意の場所で以下のコマンドを実行する
```
git clone https://github.com/o-Cytokine-o/Syskai_2019.git
```

## 3.cloneしたら、Syskai_2019/django_trump/myapp/に以下のファイルをコピーする
共有リンク：

## 4.それぞれのプログラムを実行する

### 4-1.Anacondaで作成した環境からコマンドプロンプトを立ち上げて以下のパスに移動する
```
cd Syskai_2019/django_trump
```

### 4-2.Djangoのサーバーを起動
```
python manage.py runserver
```

### 4-3.websocketサーバーを起動
```
python myapp/wsserver.py
```

## 5.GoogleChromeから以下のURLにアクセスすれば完了！
http://127.0.0.1:8000/
