import cv2
import os
from moviepy.editor import VideoFileClip
import sys
import argparse
import shutil
from v2c.config import Config
ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)  # To find local version of the library

def get_args():
    """
    Change your path here
    """
    parser = argparse.ArgumentParser(description="Classification_Branch")
    parser.add_argument("--input_video", "-i", type=str, required=False, default='/home/tuan/Documents/Code/video2command_pytorch/datasets/IIT-V2C/avi_video/P49_cam01_P49_sandwich.avi')
    parser.add_argument("--output_video", "-o", type=str, default='/home/tuan/Documents/Code/video2command_pytorch/PracticalTesting/output_video.mp4')
    parser.add_argument("--save_path", "-s", type=str, default='/home/tuan/Documents/Code/video2command_pytorch/PracticalTesting/images')
    args = parser.parse_args()
    return args

def main(args):
    if os.path.isdir(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    # Load the video
    clip = VideoFileClip(args.input_video)

    # Set the new frame rate
    new_fps = 15

    # Write the video file with the new frame rate
    clip.set_duration(clip.duration).set_fps(new_fps).write_videofile(args.output_video, codec='libx264')

    cap = cv2.VideoCapture(args.input_video)
    success, frame = cap.read()
    count = 0
    cap.set(cv2.CAP_PROP_FPS, 15)   # Force fps into 15, as stated by the author
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    while success:
        cv2.imwrite(os.path.join(args.save_path, 'frame_%d.png' % count), frame)  # Save into loseless *.png format
        count += 1
        success, frame = cap.read()

    if len(os.listdir(args.save_path)) % 30 != 0:
        integer_number = int(len(os.listdir(args.save_path)) / 30)
        bonus_frames = (integer_number + 1)*30 - len(os.listdir(args.save_path))
        for _ in range(bonus_frames):
            cv2.imwrite(os.path.join(args.save_path, 'frame_%d.png' % count), cv2.imread(os.path.join(ROOT_DIR, Config.ROOT_FOLDER, 'datasets', 'imagenet_frame.png')))  # Save into loseless *.png format
            count += 1

    print('Done.')

if __name__ == '__main__':
    args = get_args()
    main(args)