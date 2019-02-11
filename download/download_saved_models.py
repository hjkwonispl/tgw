import os
import gdown

# Make data directory
if not os.path.exists('../saved_models'):
  os.makedirs('../saved_models')
os.chdir('../saved_models')

# Download DIBCO datasets
urls = ['https://drive.google.com/uc?id=1xBlKOMWKACyxpb6aDP7wfrZNEqexKykP']
outputs = ['saved_models.tgz']

for url, output in zip(urls, outputs):
  gdown.download(url, output, quiet=False)

# Download DIBCO datasets
os.system('tar -xvf saved_models.tgz')
os.chdir('..')
