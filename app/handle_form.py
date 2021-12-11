import os

def handle_uploaded_file(f):
  os.system('rm static/media/tmdd-*.png')
  with open('media/data.csv', 'wb+') as destination:
    for chunk in f.chunks():
      destination.write(chunk)