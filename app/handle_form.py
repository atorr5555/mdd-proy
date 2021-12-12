import os

def handle_uploaded_file(f):
  os.system('rm -f static/media/tmdd-*.png')
  os.system('rm -f media/data.csv')
  with open('media/data.csv', 'wb') as destination:
    for chunk in f.chunks():
      destination.write(chunk)