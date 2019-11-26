from django.urls import path

from . import views

app_name = 'myapp'
urlpatterns = [
    path('', views.index, name='index'),
    path('game/', views.game, name='game'),
    path('view_OD/', views.view_OD, name='view_OD'),
]