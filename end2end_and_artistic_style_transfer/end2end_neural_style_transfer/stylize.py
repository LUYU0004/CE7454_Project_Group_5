"""Transform an image into the target style
Usage:
  transform.py MODEL INPUT OUTPUT [--resize=<size>]
  transform.py -h | --help
Load MODEL and use it to transform INPUT into OUTPUT.
Arguments:
  MODEL       .pth Pytorch state dict
  INPUT       input image file
  OUTPUT      output file path
Options:
  -h --help         Show this screen
  --resize=<size>   Resize shorter edge of the input [default: None]
"""
import os
import time
import sys
sys.path.insert(0,'../')

from docopt import docopt
from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import recover_image, tensor_normalizer,ImgDataset,  now_datetime
from e2e_neural_style_model.transformer_net import TransformerNet

import warnings
warnings.filterwarnings("ignore")


def load_transformer_net(model_file):
    
    print('loading model from  path << ',model_file[0])
    transformer = TransformerNet()
    transformer.load_state_dict(torch.load(model_file[0]))

    return transformer


def load_and_preprocess(image_file, size=256):
    img = Image.open(image_file).convert('RGB')
    
    transform = transforms.Compose([
            transforms.Resize([size,size]),
            transforms.ToTensor(),
            tensor_normalizer()])

    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def transform(model_file, input_path, target_path, size, kwargs):
    
    # load test dataset
    test_set = ImgDataset(input_path)
    testset_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    # load model from file 
    transformer = load_transformer_net(model_file)
    
    # mkdir for the target path
    
    target_path = target_path+now_datetime()+'/'
    os.mkdir(target_path)
    
    print('start file transformation...saving into path >>',target_path)
    with torch.no_grad():
        transformer.eval()
        
        test_set_len = len(test_set)
        for i, x in enumerate(testset_loader):
            if(i%10==0):
                print(i,' / ',test_set_len)
            img_output = transformer(x)
            output_img = Image.fromarray(
                recover_image(img_output.data.numpy())[0])
            output_img.save(target_path+test_set.getFilename(i))
    print('saved to >> ', target_path)  
    


if __name__ == "__main__":
    args = docopt(__doc__)
    print(args)
    
    kwargs ={}
    if torch.cuda.is_available():   
        torch.cuda.set_device(1)
        kwargs = {'num_workers': 4, 'pin_memory': True}
        print("GPU available >> device id: ",torch.cuda.current_device())

        
    # make sure input and output are different paths
    assert args["INPUT"] != args["OUTPUT"]
    transform(args["MODEL"], args["INPUT"], args["OUTPUT"], args["--resize"],kwargs )
