
from torchvision import transforms
from torch.utils.data import Dataset

import os
import time
from PIL import Image
import numpy as np

def now_datetime():
    '''
        return formatted date and time
    '''
    return time.strftime("%Y-%b-%d_%H-%M")

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def img_preprocessor(img_size=256):
    '''
        To preprcess input images; resize and normalize
    '''
    transform = transforms.Compose([
    transforms.Resize([img_size, img_size]), 
    #transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(), 
    tensor_normalizer()])
    
    return transform;

def tensor_normalizer():
    '''
        normalizer defined by VGG19 paper
    '''
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


def recover_image(img):
    return (
        (
            img *
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        ).transpose(0, 2, 3, 1) *
        255.
    ).clip(0, 255).astype(np.uint8)


class ImgDataset(Dataset):
    """Self generated dataset, with preprocessing.
    
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        if isinstance(root_dir, str):
            self.root_dir = root_dir
        else:
            self.root_dir = root_dir[0]
        print('created dataset with dir >> ',self.root_dir)
        
        self.fulllists = os.listdir(self.root_dir)
        self.transform = img_preprocessor()

    def __len__(self):
        
        return len(self.fulllists)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.fulllists[idx])
        image = Image.open(img_name)#io.imread(img_name)

        
        sample = self.transform(image)

        return sample
    
    def getFilename(self,idx):
        # return file name
        return self.fulllists[idx]