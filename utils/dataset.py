import os

from PIL import Image
from .transforms import get_default_transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    '''
        Initialize the dataset from given dir
    '''
    def __init__(self, dir_='./', img_size=(256,256)) -> None:
        super().__init__()
        self.dir = dir_
        self.img_size = img_size
        self.file_list = os.listdir(dir_)
        self.transform = get_default_transforms(*img_size)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dir, self.file_list[index]))
        return self.transform(img)
    
    def __len__(self):
        return len(self.file_list)

def CreateDataLoader(dir_="./datasets/real_images", img_size=(256,256), batch_size=64):
    dataset = ImageDataset(dir_, img_size)
    return DataLoader(dataset,batch_size=batch_size,shuffle=True)
