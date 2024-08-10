from torch.utils import data
import os
import v2c.utils as utils
import torch
from PIL import Image
import numpy as np


def load_annotations(dataset_path=os.path.join('video2command_pytorch', 'datasets', 'IIT-V2C'),
                     annotation_file='train.txt'):
    """Helper function to parse IIT-V2C dataset.
    """

    def get_frames_no(init_frame_no, end_frame_no):
        frames = []
        for i in range(init_frame_no, end_frame_no + 1, 1):
            frames.append(i)
        return frames

    # Read annotations
    annotations = {}
    category = ['shifting', 'cooking', 'mixing', 'cleaning', 'throwing', 'scooping', 'frying', 'closing', 'cracking',
            'turning', 'opening', 'dropping', 'cutting', 'rotating', 'spreading', 'sprinkling', 'picking', 'squeezing',
            'pushing', 'flipping', 'pulling', 'splitting', 'reaching', 'turning off', 'stirring', 'shaking', 'wasting', 'turning on',
            'melting', 'waiting', 'placing', 'holding', 'searching', 'plating', 'smearing', 'pouring', 'moving', 'changing',
            'peeling', 'grinding', 'eating']

    categories = {}
    for i, cate in enumerate(category):
        categories[cate] = i

    with open(os.path.join(dataset_path, annotation_file), 'r') as f:
        i = 0
        annotation = []
        for line in f:
            line = line.strip()
            i += 1
            annotation.append(line)

            if i % 5 == 0:
                # Classification_Branch cases
                # print(annotation)
                # assert annotation[-1] == ''
                # assert len(annotation[1].split(' ')) == 2
                # Collect Video Name, Annotated Sub-Video Clip id
                video_fname, video_id = '_'.join(annotation[0].split('_')[:-1]), annotation[0].split('_')[-1]

                # Collect init frame no ~ end frame no
                # Populate frames and commands
                init_frame_no, end_frame_no = int(annotation[1].split(' ')[0]), int(annotation[1].split(' ')[1])
                frames = get_frames_no(init_frame_no, end_frame_no)
                command = annotation[2].strip().split(' ')
                # Action
                action = [categories[annotation[3].strip()]]

                if video_fname not in annotations:
                    annotations[video_fname] = [[video_id, frames, command, action]]
                else:
                    annotations[video_fname].append([video_id, frames, command, action])

                annotation = []

    return annotations



def clipsname_captions(annotations):
    """Get (clip_name, target) pairs from annotation.
    """
    # Parse all (inputs, captions) pair
    clips_name, actions = [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get command caption
            action = annotation[3][0]

            actions.append(action)
            clips_name.append(clip_name)
    # print(actions)
    return clips_name, actions



def imgspath_targets_v1(annotations,
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png'),
                        padding_words=True):
    """Get training/test image-command pairs.
    Version v2 strategy: Same as original IIT-V2C strategy, have a number
    of max_frames images per sample. Cut images larger than max_frames, populate
    sample if no. images is smaller than max_frames.
    """

    def get_frames_path(frames_no,
                        video_fname,
                        max_frames=30,
                        dataset_path=os.path.join('datasets', 'IIT-V2C'),
                        folder='images',
                        synthetic_frame_path=os.path.join('imagenet_frame.png')):
        """Helper func to parse image path from numbers.
        """
        # Cut additional images by getting min loop factor
        num_frames = len(frames_no)
        loop_factor = min(num_frames, max_frames)

        imgs_path = []
        for i in range(loop_factor):
            img_path = os.path.join(dataset_path, folder, video_fname, 'frame{}.png'.format(frames_no[i]))
            # print(img_path)
            # print(os.path.isfile(img_path))
            if os.path.isfile(img_path):  # Check if frame exists
                imgs_path.append(img_path)

        # Add synthetically made imagenet frame
        while len(imgs_path) < max_frames:
            imgs_path.append(synthetic_frame_path)
        # print(imgs_path)
        return imgs_path

    # Parse all (inputs, targets) pair
    inputs, targets = [], []
    for video_fname in annotations.keys():
        annotations_by_clip = annotations[video_fname]
        for annotation in annotations_by_clip:
            # Clip name
            clip_name = video_fname + '_' + annotation[0]

            # Get all images of the current clip
            frames_path = get_frames_path(annotation[1],
                                          video_fname,
                                          max_frames,
                                          dataset_path,
                                          folder,
                                          synthetic_frame_path)

            # Get command caption
            action = annotation[3][0]

            inputs.append({clip_name: frames_path})
            targets.append(action)

    return inputs, targets


# ----------------------------------------
# Functions for torch.data.Dataset
# ----------------------------------------

def parse_dataset(config,
                  annotation_file,
                  numpy_features=True):
    """Parse IIT-V2C dataset and update configuration.
    """

    # Load annotation 1st
    annotations = load_annotations(config.DATASET_PATH, annotation_file)

    # Use images
    if numpy_features:
        clips, actions = clipsname_captions(annotations)
        clips = [os.path.join(config.DATASET_PATH, list(config.BACKBONE.keys())[0], x + '.npy') for x in clips]

    # Use images
    else:
        clips, actions = imgspath_targets_v1(annotations,
                                              max_frames=config.WINDOW_SIZE,
                                              dataset_path=config.DATASET_PATH,
                                              folder='images',
                                              synthetic_frame_path=os.path.join(config.ROOT_DIR, 'datasets',
                                                                                'imagenet_frame.png')
                                              )

    return clips, actions, config

class FeatureDataset(data.Dataset):
    """Create an instance of IIT-V2C dataset with (features, targets) pre-extracted,
    or with (imgs_path, targets)
    """
    def __init__(self,
                 inputs,
                 actions,
                 numpy_features=True,
                 transform=None):
        self.inputs, self.targets = inputs, actions     # Load annotations
        self.numpy_features = numpy_features
        self.transform = transform

    def parse_clip(self,
                   clip):
        """Helper function to parse images {clip_name: imgs_path} into a clip.
        """
        Xv = []
        clip_name = list(clip.keys())[0]
        imgs_path = clip[clip_name]
        for img_path in imgs_path:
            img = self._imread(img_path)
            Xv.append(img)

        Xv = torch.stack(Xv, dim=0)
        return Xv, clip_name

    def _imread(self, path):
        """Helper function to read image.
        """
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        if self.numpy_features:
            Xv = np.load(self.inputs[idx])
            clip_name = self.inputs[idx].split('/')[-1]
            S = self.targets[idx]
        # Image dataset
        else:
            Xv, clip_name = self.parse_clip(self.inputs[idx])
            S = self.targets[idx]

        return Xv, S, clip_name

