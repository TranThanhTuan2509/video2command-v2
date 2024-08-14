## Video Understanding with Video2Command

A PyTorch adapted implementation of the video-to-command model described in the paper:

"[Sequence to Sequence – Video to Text](https://arxiv.org/pdf/1505.00487)".

"[Translating Videos to Commands for Robotic Manipulation with Deep Recurrent Neural Networks](https://sites.google.com/site/video2command/)".

"[V2CNet: A Deep Learning Framework to Translate Videos to Commands for Robotic Manipulation](https://arxiv.org/pdf/1903.10869)".

"[Watch and Act: Learning Robotic Manipulation From Visual Demonstration](https://www.researchgate.net/publication/369127059_Watch_and_Act_Learning_Robotic_Manipulation_From_Visual_Demonstration)".

`This model was revised by Thanh Tuan in 2024`

- Using both CNNs and RNNs are not enough to understand captured actions and interacted objects. According to `Watch and Act`, a new model was proposed some new methods for video understanding problems.

## Requirements
- You first create a new Anaconda environment:

      conda create -n c2v python=3.12.4
- Activate the new environment using:

      conda activate c2v
- Install all required libraries with:

      pip install -r requirements.txt

## Introduction
- The *video2command* model is an Encoder-Decoder neural network that learns to generate a short sentence which can be used to command a robot to perform various manipulation tasks. The architecture of the network is listed below:
  
- The left side branch uses Mask-RCNN to isolate unimportance and irrelevance objects in that scene by subtracting 2 frames (the current frame and the next frame) and then fedding it through a CNN to extract the feature.
  
- In most imitation testing cases in practice, the robot's vision comes from a fixed camera. Based on this, we know that the object's size that the robot has to handle will be less than 0.55 or 0.6 of the captured image. The robot will interact with the object that has the highest confidence score in object detection and the highest probability of being a pickable object.

<p align="center">
  <picture>
    <img alt="image" src="https://github.com/TranThanhTuan2509/video2command-v2/blob/main/images/Translation.png "video2command"" width="600" height="350" style="max-width: 100%;">
  </picture>
</p>

- The new branch was adding to run simultaneously with the translation one. The main function of this branch is action recognition

<p align="center">
  <picture>
    <img alt="image" src="https://github.com/TranThanhTuan2509/video2command-v2/blob/main/images/Classification.png "video2command"" width="600" height="350" style="max-width: 100%;">
  </picture>
</p>

## Experiment
To repeat the *video2command* experiment:
1. Clone the repository.

2. Download the [IIT-V2C dataset](https://sites.google.com/site/iitv2c/), extract the dataset and setup the directory path as `datasets/IIT-V2C`.

4. For CNN features, two options are provided:
     - Use the [pre-extracted ResNet50 features](https://drive.google.com/file/d/1Y_YKHB4Bw6MPXj05S36d1G_3rMx73Uv5/view?usp=sharing) provided by the original author.

     - Perform feature extraction yourself. Firstly run `avi2frames.py` under folder `experiments/experiment_IIT-V2C` to convert all videos into images. Download the [*.pth weights for ResNet50](https://github.com/ruotianluo/pytorch-resnet) converted from Caffe. Run `extract_features.py` under folder `experiments/experiment_IIT-V2C` afterwards.
       
     - Download the " " for Mask-RCNN pretrained checkpoint.
     
     - Note that the author's pre-extracted features seem to have a better quality and lead to a possible 1~2% higher metric scores.

5. To begin training, run `train_iit-v2c.py`.
   - NOTE: You need more than 100GB free space for this process if you choose training from scratch with `IIT-V2C` dataset.

7. For evaluation, firstly run `evaluate_iit-v2c.py` to generate predictions given all saved checkpoints. Run `cocoeval_iit-v2c.py` to calculate scores for the predictions.
      - In case, Java is not existed in your environment. Open your terminal and run these following lines:
        
      - For Ubuntu/Debian:
        
              sudo apt update
        
              sudo apt install default-jre
   
      - For macOS:
        
              brew update
        
              brew install openjdk
        
      - If Java is installed but not in your PATH, you can add it. On Linux or macOS, you can add the following lines to your ~/.bashrc or ~/.zshrc file:
        
              export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
        
              export PATH=$JAVA_HOME/bin:$PATH
8. For practical testing, access `PracticalTesting` folder and run `video->frame.py`, `video extraction.py`, `evaluation.py` step by step, don't forget to change your directory path

# Contact
If you have any questions or comments, please send an to `22023506@vnu.edu.vn`

