import os.path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import Classification
from tqdm.autonotebook import tqdm
import argparse
import shutil
from Classification_Branch import dataset
from torch.utils import data
from v2c.config import *
import sys
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.abspath("../../")

# Import v2c utils
sys.path.append(ROOT_DIR)


def get_args():
    parser = argparse.ArgumentParser(description="Train a CNN model")
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--num_workers", "-w", type=int, default=4)
    parser.add_argument("--log_path", "-p", type=str, default="action_tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="action_checkpoints")
    parser.add_argument("--checkpoint_model", "-m", type=str, default=None)
    parser.add_argument("--lr", "-l", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def train(args):
    class TrainConfig(Config):
        """Configuration for training with IIT-V2C.
        """
        NAME = 'v2c_IIT-V2C'
        ROOT_DIR = ROOT_DIR
        CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'checkpoints')
        DATASET_PATH = os.path.join(ROOT_DIR, 'video2command_pytorch', 'datasets', 'IIT-V2C')

    # Setup configuration class
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_annotation_file = 'train.txt'
    clips, targets, config = dataset.parse_dataset(config, train_annotation_file)
    config.display()
    train_dataset = dataset.FeatureDataset(clips, targets)
    train_dataloader = data.DataLoader(train_dataset,
                                   batch_size=config.BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=config.WORKERS)

    test_annotation_file = 'test.txt'
    clips, targets, config = dataset.parse_dataset(config, test_annotation_file)
    config.display()
    test_dataset = dataset.FeatureDataset(clips, targets)
    test_dataloader = data.DataLoader(test_dataset,
                                       batch_size=config.BATCH_SIZE,
                                       shuffle=False,
                                       num_workers=config.WORKERS)

    model = Classification(num_classes=41).to(device)
    # summary(model, input_size=(1, 3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.checkpoint_model and os.path.isfile(args.checkpoint_model):
        checkpoint = torch.load(args.checkpoint_model)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_loss = checkpoint["best_loss"]
    else:
        start_epoch = 0
        best_loss = 1000

    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if os.path.isdir(args.checkpoint_path):
        shutil.rmtree(args.checkpoint_path)
    os.makedirs(args.checkpoint_path)
    writer = SummaryWriter(args.log_path)

    for epoch in range(start_epoch, args.epochs):
        # MODEL TRAINING
        model.train()
        progress_bar = tqdm(train_dataloader, colour="blue")
        for iter, (images, labels, clip_name) in enumerate(progress_bar):
            # print(images)
            # print(labels)
            # print(clip_name)
            images = images.to(device)
            labels = labels.to(device).long()
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward pass + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch + 1, args.epochs, loss.item()))
            writer.add_scalar("Train/loss", loss.item(), iter + epoch * len(train_dataloader))

        # MODEL VALIDATION
        all_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader, colour="yellow")
            for iter, (images, labels, clip_name) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device).long()
                # Forward pass
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                all_losses.append(loss.item())

            acc = accuracy_score(all_labels, all_predictions)
            loss = sum(all_losses) / len(all_losses)
            print("Epoch {}. Validation loss: {}. Validation accuracy: {}".format(epoch + 1, loss, acc))
            writer.add_scalar("Valid/loss", loss, epoch)
            writer.add_scalar("Valid/acc", acc, epoch)

        # save model
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))
        if loss < best_loss:
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
            best_loss = loss


if __name__ == '__main__':
    args = get_args()
    train(args)
