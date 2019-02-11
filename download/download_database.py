import os
import gdown

# Make data directory
if not os.path.exists('../data/Adience'):
  os.makedirs('../data/Adience')
os.chdir('../data/Adience')

# Download DIBCO datasets
urls = ['https://drive.google.com/uc?id=1du0sEkpKfL1EJeP70oRB45ZKODQ8vbID',
  'https://drive.google.com/uc?id=1fcFw4uX9mUrKjhys1G6JY9zxxDuPtc0C',
  'https://drive.google.com/uc?id=1HN2JVjWdLKG9avlOhp79L_R__EZ-u7xJ']
outputs = ['age_train.tgz',
  'age_val.tgz',
  'age_test.tgz']

for url, output in zip(urls, outputs):
  gdown.download(url, output, quiet=False)

# Download DIBCO datasets
os.system('tar -xvf age_train.tgz')
os.system('tar -xvf age_val.tgz')
os.system('tar -xvf age_test.tgz')
os.chdir('..')
