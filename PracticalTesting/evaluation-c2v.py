import os
import glob
import sys
import pickle

import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library
from v2c.model import *
from v2c import utils
from v2c.config import *
from datasets import iit_v2c

# Configuration for hperparameters
class TestConfig(Config):
    """Configuration for training with IIT-V2C.
    """
    NAME = 'v2c_IIT-V2C'
    ROOT_DIR = ROOT_DIR
    DATASET_PATH = os.path.join(ROOT_DIR, 'datasets', 'IIT-V2C')
    MAXLEN = 10

# Setup configuration class
config = TestConfig()
# Setup tf.dataset objectimages
vocab = pickle.load(open(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'checkpoints', 'vocab.pkl'), 'rb'))
clips = sorted(os.listdir(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'PracticalTesting', list(config.BACKBONE.keys())[0])),
               key = lambda x: int(x.split('.')[0].split('_')[-1]), reverse = False)
clips = [os.path.join(ROOT_DIR, Config.ROOT_FOLDER, 'PracticalTesting', list(config.BACKBONE.keys())[0], x) for x in clips]
test_dataset = iit_v2c.PracticalFeatureDataset(clips)
test_loader = data.DataLoader(test_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.WORKERS)

config.display()

# Setup and build video2command training inference
v2c_model = Video2Command(config)
v2c_model.build()

# Safely create prediction dir if non-exist
if not os.path.exists(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'PracticalTesting', 'prediction')):
    os.makedirs(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'PracticalTesting', 'prediction'))

# Start evaluating
checkpoint_files = sorted(glob.glob(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'checkpoints', 'saved', '*.pth')))
for checkpoint_file in checkpoint_files:
    epoch = int(checkpoint_file.split('_')[-1][:-4])
    v2c_model.load_weights(checkpoint_file)
    y_pred, action = v2c_model.evaluate(test_loader, vocab, practical=True)

    # Save to evaluation file
    f = open(os.path.join(ROOT_DIR, config.ROOT_FOLDER, 'PracticalTesting', 'prediction',
                          'prediction_{}.txt'.format(epoch)), 'w')

    for i, (y, a) in enumerate(zip(y_pred, action)):
        pred_command = utils.sequence_to_text(y, vocab)
        f.write('------------------------------------------\n')
        f.write(f'action_{i + 1}' + '\n')
        f.write(pred_command + '\n')
        f.write(a + '\n')

    print('Ready for cococaption.')