import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
import datasets.iit_v2c as iit_v2c
from v2c.config import *
from v2c.model import *

# Configuration for hperparameters
class FEConfig(Config):
    """Configuration for feature extraction.
    """
    NAME = 'Feature_Extraction'
    MODE = 'test'
    ROOT_DIR = ROOT_DIR
    WINDOW_SIZE = 30
    BATCH_SIZE = 50

def extract(dataset,
            model_name):
    # Create output directory
    dataset_path = os.path.join(ROOT_DIR, Config.ROOT_FOLDER, 'PracticalTesting')
    output_path = os.path.join(dataset_path, model_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare pre-trained model
    print('Loading pre-trained model...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNNWrapper(backbone=model_name,
                       checkpoint_path=os.path.join(ROOT_DIR, Config.ROOT_FOLDER, 'checkpoints', 'backbone', 'resnet50.pth'))
    model.eval()
    model.to(device)
    print('Done loading.')

    # Feature extraction
    # for i, (Xv, mask, clip_name) in enumerate(dataset):
    for i, (Xv, clip_name) in enumerate(dataset):
    # print(type(Xv))
        with torch.no_grad():
            Xv = Xv.to(device)
            mask = mask.to(device)
            # print(mask.dtype)
            print('-' * 30)
            print('Processing clip {}...'.format(clip_name))
            # print(imgs_path, clip_name)
            # assert len(imgs_path) == 30

            # output features of original images
            Xv_outputs = model(Xv)
            Xv_outputs = Xv_outputs.view(Xv_outputs.shape[0], -1)
            print(Xv_outputs.shape)

            # output features of mask images
            mask_outputs = model(mask)
            mask_outputs = mask_outputs.view(mask_outputs.shape[0], -1)
            print(mask_outputs.shape)
            outputs = torch.cat((Xv_outputs, mask_outputs), dim=0)

            # Save into clips
            outfile_path = os.path.join(output_path, clip_name+'.npy')
            np.save(outfile_path, outputs.cpu().numpy())
            print('Shape: {}, saved to {}.'.format(outputs.shape, outfile_path))
    del model
    return

def main_iit_v2c():
    # Parameters
    config = FEConfig()
    model_names = ['resnet50']

    # Get torch.dataset object
    videos = [os.path.join(ROOT_DIR, Config.ROOT_FOLDER, 'PracticalTesting', 'images', f'{frame}')
              for frame in
              os.listdir(os.path.join(ROOT_DIR, 'video2command_pytorch', 'PracticalTesting', 'images'))]

    iteration = int(len(videos) / 30)
    clips = []
    for action_num in range(iteration):
        clips.append({f'action_{action_num + 1}': videos[action_num: action_num + 30]})

    config.display()
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image_dataset = iit_v2c.PracticalFeatureDataset(clips,
                                           numpy_features=False,
                                           transform=transform)
    for model_name in model_names:
        extract(image_dataset, model_name)


if __name__ == '__main__':
    main_iit_v2c()
