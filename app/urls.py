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
  path('column-list', views.column),
  path('clustering/process', views.clustering_process),
  path('column-listk', views.column_kmeans),
  path('kmeans/', views.kmeans),
  path('kmeans/process', views.kmeans_process),
]