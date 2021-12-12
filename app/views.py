from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect

from .forms import UploadFileForm

import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from apyori import apriori

# Imaginary function to handle an uploaded file.
from .handle_form import handle_uploaded_file

# Create your views here.
def index(request):
  os.system('rm -f static/media/tmdd-*.png')
  #return HttpResponse('Hello')
  return render(request, 'index.html')

def eda(request):
  os.system('rm -f static/media/tmdd-*.png')
  form = UploadFileForm()
  return render(request, 'eda.html', {'form': form})

# Procesamiento para tratar la eda
def eda_process(request):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
    if not os.path.exists('media/data.csv'):
      return HttpResponseRedirect('/app/eda')
    dict_params = {}
    # Leer datos
    datos = pd.read_csv('media/data.csv')
    table = datos.head().to_html()
    dict_params['table'] = table
    #Dimensiones de los datos
    shape = datos.shape
    dict_params['shape'] = shape
    # Tipos de datos
    data_types_se = datos.dtypes
    data_types_df = data_types_se.to_frame()
    data_types_df.columns = ['Tipo']
    data_types = data_types_df.to_html()
    dict_params['data_types'] = data_types
    # Datos faltantes
    faltantes_se = datos.isnull().sum()
    faltantes_df = faltantes_se.to_frame()
    faltantes_df.columns = ['Número de datos faltantes']
    faltantes = faltantes_df.to_html()
    dict_params['faltantes'] = faltantes
    # Histogramas
    plt.figure()
    datos.hist(figsize=(14,14), xrot=45)
    plt.savefig('static/media/tmdd-hist-num.png')
    # Resumen estadistico
    res_estadistico_df = datos.describe()
    res_estadistico = res_estadistico_df.to_html()
    dict_params['res_estadistico'] = res_estadistico
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
    dict_params['list_boxplots'] = list_boxplots
    # Resumen estadistico de variables no numericas
    res_estadistico_cat = ''
    try:
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
      dict_params['res_estadistico_cat'] = res_estadistico_cat
      dict_params['list_countplot'] = list_countplot
    except:
      res_estadistico_cat = False
    # Mapa de calor de correlaciones
    plt.figure(figsize=(14,7))
    sns.heatmap(datos.corr(), cmap='RdBu_r', annot=True)
    plt.savefig('static/media/tmdd-corr.png')

    return render(request, 'eda-processed.html', dict_params)
  else:
    return HttpResponseRedirect('/app/eda/')

def acd(request):
  os.system('rm -f static/media/tmdd-*.png')
  form = UploadFileForm()
  return render(request, 'acd.html', {'form': form})

# Procesamiento para tratar la acd
def acd_process(request):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
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
  else:
    return HttpResponseRedirect('/app/acd/')

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
    datos.to_csv(index=False, path_or_buf=response)  # with other applicable parameters
    return response

def pca(request):
  os.system('rm -f static/media/tmdd-*.png')
  form = UploadFileForm()
  return render(request, 'pca.html', {'form': form})

# Procesamiento para tratar la pca
def pca_process(request):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
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
  else:
    return HttpResponseRedirect('/app/pca/')

def clustering(request):
  os.system('rm -f static/media/tmdd-*.png')
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
      columnas_datos = datos.select_dtypes([np.number]).columns
      datos = datos[columnas_datos]
      # Estandarizar datos
      Matriz = np.array(datos)
      estandarizar = StandardScaler()
      MEstandarizada = estandarizar.fit_transform(Matriz)
      table_est = pd.DataFrame(MEstandarizada).head().to_html()
      # Gracifa de arbol
      plt.figure(figsize=(10, 7))
      plt.ylabel('Distancia')
      Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
      plt.savefig('static/media/tmdd-arbol.png')
    else:
      print('invalid form')
      print(form.errors)
  return HttpResponse(columnas_texto)

def clustering_process(request):
  if request.method == 'POST':
    if not os.path.exists('media/data.csv'):
      return HttpResponseRedirect('/app/pca')
    # Leer datos
    datos = pd.read_csv('media/data.csv')
    dicti = {}
    for x in datos.columns:
      dicti[x] = bool(request.POST.get(x, 0))
    new_cols = []
    for key in dicti:
      if dicti[key]:
        new_cols.append(key)
    datos = datos[new_cols]
    n_clustersv = int(request.POST.get('n_clusters', 0))
    columnas_datos = datos.select_dtypes([np.number]).columns
    datos = datos[columnas_datos]
    # Estandarizar datos
    Matriz = np.array(datos)
    estandarizar = StandardScaler()
    MEstandarizada = estandarizar.fit_transform(Matriz)
    table_est = pd.DataFrame(MEstandarizada).head().to_html()
    # Gracifa de arbol
    plt.figure(figsize=(10, 7))
    plt.ylabel('Distancia')
    Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
    plt.savefig('static/media/tmdd-arbol.png')
    #Se crean las etiquetas de los elementos en los clústeres
    MJerarquico = AgglomerativeClustering(n_clusters=n_clustersv, linkage='complete', affinity='euclidean')
    MJerarquico.fit_predict(MEstandarizada)
    datos['clusterH'] = MJerarquico.labels_
    CentroidesH = datos.groupby('clusterH').mean()
    centroides = CentroidesH.to_html()

    return render(request, 'clustering-processed.html', {'table_est': table_est,
                                                        'centroides': centroides})
  else:
    return HttpResponseRedirect('/app/clustering/')

def kmeans(request):
  os.system('rm -f static/media/tmdd-*.png')
  form = UploadFileForm()
  return render(request, 'kmeans.html', {'form': form})

def column_kmeans(request):
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

def kmeans_process(request):
  if request.method == 'POST':
    if not os.path.exists('media/data.csv'):
      return HttpResponseRedirect('/app/pca')
    # Leer datos
    datos = pd.read_csv('media/data.csv')
    dicti = {}
    for x in datos.columns:
      dicti[x] = bool(request.POST.get(x, 0))
    new_cols = []
    for key in dicti:
      if dicti[key]:
        new_cols.append(key)
    datos = datos[new_cols]
    columnas_datos = datos.select_dtypes([np.number]).columns
    datos = datos[columnas_datos]
    # Estandarizar datos
    Matriz = np.array(datos)
    estandarizar = StandardScaler()
    MEstandarizada = estandarizar.fit_transform(Matriz)
    table_est = pd.DataFrame(MEstandarizada).head().to_html()
    # Encontrar cuantos clusters
    SSE = []
    for i in range(2, 12):
      km = KMeans(n_clusters=i, random_state=0)
      km.fit(MEstandarizada)
      SSE.append(km.inertia_)
    #Se grafica SSE en función de k
    plt.figure(figsize=(10, 7))
    plt.plot(range(2, 12), SSE, marker='o')
    plt.xlabel('Cantidad de clusters *k*')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
    n_clustersv = kl.elbow
    plt.style.use('ggplot')
    kl.plot_knee()
    plt.savefig('static/media/tmdd-codo.png')
    #Se crean las etiquetas de los elementos en los clusters
    MParticional = KMeans(n_clusters=n_clustersv, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    datos['clusterP'] = MParticional.labels_
    CentroidesH = datos.groupby('clusterP').mean()
    centroides = CentroidesH.to_html()


    return render(request, 'kmeans-processed.html', {'table_est': table_est,
                                                        'centroides': centroides})
  else:
    return HttpResponseRedirect('/app/kmeans/')

def assoc(request):
  os.system('rm -f static/media/tmdd-*.png')
  form = UploadFileForm()
  return render(request, 'assoc.html', {'form': form})

# Procesamiento para tratar la eda
def assoc_process(request):
  if request.method == 'POST':
    form = UploadFileForm(request.POST, request.FILES)
    if form.is_valid():
      handle_uploaded_file(request.FILES['file'])
    else:
      print(form.errors)
    if not os.path.exists('media/data.csv'):
      return HttpResponseRedirect('/app/assoc')
    # Leer datos
    datos = pd.read_csv('media/data.csv', header=None)
    table = datos.head().to_html()
    #Se incluyen todas las transacciones en una sola lista
    Transacciones = datos.values.reshape(-1).tolist()
    #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
    ListaM = pd.DataFrame(Transacciones)
    ListaM['Frecuencia'] = 0
    #Se agrupa los elementos
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
    ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
    ListaM = ListaM.rename(columns={0 : 'Item'})
    # Se genera un gráfico de barras
    plt.figure(figsize=(16,20), dpi=300)
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(ListaM['Item'], width=ListaM['Frecuencia'], color='blue')
    plt.savefig('static/media/tmdd-freq.png')

    MoviesLista = datos.stack().groupby(level=0).apply(list).tolist()

    soporte = float(request.POST.get('soporte', 0))
    confianza = float(request.POST.get('confianza', 0))
    elevacion = float(request.POST.get('elevacion', 0))
    ReglasC1 = apriori(MoviesLista, 
                    min_support=soporte, 
                    min_confidence=confianza, 
                    min_lift=elevacion)
    ResultadosC1 = list(ReglasC1)
    df = pd.DataFrame(ResultadosC1)
    df['confidence'] = df.apply(lambda row : row['ordered_statistics'][0][2], axis = 1)
    df['lift'] = df.apply(lambda row : row['ordered_statistics'][0][3], axis = 1)
    df = df.drop(columns=['ordered_statistics'])
    reglas = df.to_html()

    return render(request, 'assoc-processed.html', {'table': table,
                                                    'reglas': reglas})
  else:
    return HttpResponseRedirect('/app/assoc/')
