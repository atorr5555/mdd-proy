import os

def handle_uploaded_file(f, h):
  os.system('rm -f static/media/tmdd-*' + str(h) + '.png')
  os.system('rm -f media/data*' + str(h) + '.csv')
  with open('media/data' + str(h) + '.csv', 'wb') as destination:
    for chunk in f.chunks():
      destination.write(chunk)


def handle_uploaded_file2(f, h):
  os.system('rm -f static/media/tmdd-*' + str(h) + '.png')
  os.system('rm -f media/data*' + str(h) + '.csv')
  with open('media/data-ad' + str(h) + '.csv', 'wb') as destination:
    for chunk in f.chunks():
      destination.write(chunk)