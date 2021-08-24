import torch
import warnings
import gdown
import os.path

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = 'lungmask/lung_seg_model.pth'
if not os.path.exists(file_path):
    url = 'https://drive.google.com/uc?id=1CHatlZDtYbpFoUOE7XhILVIzBJXkp-nS'
    output = 'lungmask/lung_seg_model.pth'
    print('downloading lung model...')
    gdown.download(url, output, quiet=False)
model = torch.load(file_path, map_location=device)
model.eval()
print('lung model loaded...')