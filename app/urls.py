from django.urls import path
from . import views

#URLConf
urlpatterns = [
  path('', views.index),
  path('eda/', views.eda),
  path('eda/process', views.eda_process),
  path('acd/', views.acd),
  path('acd/process', views.acd_process),
  path('download/', views.download),
  path('pca/', views.pca),
  path('pca/process', views.pca_process),
  path('clustering/', views.clustering),
  path('column-list', views.column)
]