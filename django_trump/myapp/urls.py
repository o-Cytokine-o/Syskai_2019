from django.urls import path

from . import views

app_name = 'myapp'
urlpatterns = [
    path('', views.index, name='index'),
    path('game/', views.game, name='game'),
    path('game2/', views.game2, name='game2'),
    path('game_2player/', views.game_2player, name='game_2player'),
    path('view_OD/', views.view_OD, name='view_OD'),
    path('view_OD_2player/', views.view_OD_2player, name='view_OD_2player'),
    path('view_OD_no_tut/', views.view_OD_no_tut, name='view_OD_no_tut'),
]