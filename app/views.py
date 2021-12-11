from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

from .forms import UploadFileForm

import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Imaginary function to handle an uploaded file.
from .handle_form import handle_uploaded_file

# Create your views here.
def index(request):
  os.system('rm static/media/tmdd-*.png')
  #return HttpResponse('Hello')
  return render(request, 'index.html')

def eda(request):
  os.system('rm static/media/tmdd-*.png')
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
      return HttpResponseRedirect('/app/eda/process')
  else:
    form = UploadFileForm()
  return render(request, 'eda.html', {'form': form})

# Procesamiento para tratar la eda
def eda_process(request):
  if not os.path.exists('media/data.csv'):
    return HttpResponseRedirect('/app/eda')
  # Leer datos
  datos = pd.read_csv('media/data.csv')
  table = datos.head().to_html()
  #Dimensiones de los datos
  shape = datos.shape
  # Tipos de datos
  data_types_se = datos.dtypes
  data_types_df = data_types_se.to_frame()
  data_types_df.columns = ['Tipo']
  data_types = data_types_df.to_html()
  # Datos faltantes
  faltantes_se = datos.isnull().sum()
  faltantes_df = faltantes_se.to_frame()
  faltantes_df.columns = ['Número de datos faltantes']
  faltantes = faltantes_df.to_html()
  # Histogramas
  plt.figure()
  datos.hist(figsize=(14,14), xrot=45)
  plt.savefig('static/media/tmdd-hist-num.png')
  # Resumen estadistico
  res_estadistico_df = datos.describe()
  res_estadistico = res_estadistico_df.to_html()
  # Boxplots
  i = 0
  list_boxplots = []
  for col in datos.select_dtypes([np.number]).columns:
    plt.figure()
    sns.boxplot(col, data=datos)
    name = 'static/media/tmdd-boxplot' + str(i) + '.png'
    i += 1
    plt.savefig(name)
    list_boxplots.append(name)
  # Resumen estadistico de variables no numericas
  res_estadistico_cat_df = datos.describe(include='object')
  res_estadistico_cat = res_estadistico_cat_df.to_html()
  # Graficas de variables no numericas
  i = 0
  list_countplot = []
  for col in datos.select_dtypes(include='object'):
    if datos[col].nunique()<34:
      plt.figure()
      sns.countplot(y=col, data=datos)
      name = 'static/media/tmdd-countplot' + str(i) + '.png'
      i += 1
      plt.savefig(name)
      list_countplot.append(name)
  # Mapa de calor de correlaciones
  plt.figure(figsize=(14,7))
  sns.heatmap(datos.corr(), cmap='RdBu_r', annot=True)
  plt.savefig('static/media/tmdd-corr.png')

  return render(request, 'eda-processed.html', {'table': table,
                                                'shape': shape,
                                                'data_types': data_types,
                                                'faltantes': faltantes,
                                                'res_estadistico': res_estadistico,
                                                'list_boxplots': list_boxplots,
                                                'res_estadistico_cat': res_estadistico_cat,
                                                'list_countplot': list_countplot})

def acd(request):
  os.system('rm static/media/tmdd-*.png')
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
      return HttpResponseRedirect('/app/acd/process')
  else:
    form = UploadFileForm()
  return render(request, 'acd.html', {'form': form})

# Procesamiento para tratar la acd
def acd_process(request):
  if not os.path.exists('media/data.csv'):
    return HttpResponseRedirect('/app/acd')
  # Leer datos
  datos = pd.read_csv('media/data.csv')
  # Tipos de datos
  data_types_se = datos.dtypes
  data_types_df = data_types_se.to_frame()
  data_types_df.columns = ['Tipo']
  data_types = data_types_df.to_html()
  # Datos faltantes
  faltantes_se = datos.isnull().sum()
  faltantes_df = faltantes_se.to_frame()
  faltantes_df.columns = ['Número de datos faltantes']
  faltantes = faltantes_df.to_html()
  # Evaluacion visual
  plt.figure()
  sns.pairplot(datos)
  plt.savefig('static/media/tmdd-eval-visual.png')
  # Matriz de correlaciones
  plt.figure(figsize=(14,7))
  MatrizInf = np.triu(datos.corr())
  sns.heatmap(datos.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
  plt.savefig('static/media/tmdd-corr.png')
  # Columnas para seleccion
  columnas = datos.columns
  return render(request, 'acd-processed.html', {'data_types': data_types,
                                                'faltantes': faltantes,
                                                'columnas': columnas})

# Procesamiento para descargar un nuevo dataset
def download(request):
  if request.method != 'POST':
    return HttpResponseRedirect('/app/')
  # Leer datos
  datos = pd.read_csv('media/data.csv')
  if request.method == 'POST':
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=export.csv'
    dicti = {}
    for x in datos.columns:
      dicti[x] = bool(request.POST.get(x, 0))
    new_cols = []
    for key in dicti:
      if dicti[key]:
        new_cols.append(key)
    datos = datos[new_cols]
    datos.to_csv(path_or_buf=response)  # with other applicable parameters
    return response

def pca(request):
  os.system('rm static/media/tmdd-*.png')
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
      return HttpResponseRedirect('/app/pca/process')
  else:
    form = UploadFileForm()
  return render(request, 'pca.html', {'form': form})

# Procesamiento para tratar la pca
def pca_process(request):
  if not os.path.exists('media/data.csv'):
    return HttpResponseRedirect('/app/pca')
  # Leer datos
  datos = pd.read_csv('media/data.csv')
  columnas_datos = datos.select_dtypes([np.number]).columns
  datos = datos[columnas_datos]
  # Datos normalizados
  normalizar = StandardScaler()                      # Se instancia el objeto StandardScaler 
  normalizar.fit(datos)                           # Se calcula la media y desviación para cada variable
  MNormalizada = normalizar.transform(datos)      # Se normalizan los datos 
  pca = PCA(n_components=10)             # Se instancia el objeto PCA    #pca=PCA(n_components=None), pca=PCA(.85)
  pca.fit(MNormalizada)                  # Se obtiene los componentes
  componentes_df = pd.DataFrame(pca.components_)
  componentes = componentes_df.to_html()
  # Grafica de varianza acumulada
  plt.figure()
  plt.plot(np.cumsum(pca.explained_variance_ratio_))
  plt.xlabel('Número de componentes')
  plt.ylabel('Varianza acumulada')
  plt.grid()
  plt.savefig('static/media/tmdd-varacum.png')
  #Cargas de componentes
  CargasComponentes = pd.DataFrame(abs(pca.components_), columns=columnas_datos)
  cargas = CargasComponentes.to_html()
  
  # Columnas para seleccion
  columnas = datos.columns
  return render(request, 'pca-processed.html', {'columnas': columnas,
                                                'componentes': componentes,
                                                'cargas': cargas})

def clustering(request):
  os.system('rm static/media/tmdd-*.png')
  if request.method == 'POST':
    datos = pd.read_csv('media/data.csv')
    dicti = {}
    for x in datos.columns:
      dicti[x] = bool(request.POST.get(x, 0))
    new_cols = []
    for key in dicti:
      if dicti[key]:
        new_cols.append(key)
    datos = datos[new_cols]
    datos.to_csv(path_or_buf='media/data.csv')
    return HttpResponseRedirect('/app/clustering/process')
  else:
    form = UploadFileForm()
  return render(request, 'clustering.html', {'form': form})

def column(request):
  if request.method == 'POST':
    form = UploadFileForm(data=request.POST, files=request.FILES)
    if form.is_valid():
      print('valid form')
      handle_uploaded_file(request.FILES['file'])
      datos = pd.read_csv('media/data.csv')
      columnas = datos.columns
      columnas_texto = ",".join(columnas)
    else:
      print('invalid form')
      print(form.errors)
  return HttpResponse(columnas_texto)