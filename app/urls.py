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
  path('assoc/', views.assoc),
  path('assoc/process', views.assoc_process),
  path('adpro/', views.adpro),
  path('column-listadpro', views.column_adpro),
  path('adpro/process', views.adpro_process),
  path('column-listadclas', views.column_adclas),
  path('adclas/', views.adclas),
  path('adclas/process', views.adclas_process),
  path('adpro/download', views.download_adpro),
  path('adclas/download', views.download_adclas),
]