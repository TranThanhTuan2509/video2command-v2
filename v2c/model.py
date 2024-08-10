import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from v2c.backbone import resnet


# ----------------------------------------
# Functions for Video Feature Extraction
# ----------------------------------------

class CNNWrapper(nn.Module):
    """Wrapper module to extract features from image using
    pre-trained CNN.
    """

    def __init__(self,
                 backbone,
                 checkpoint_path):
        super(CNNWrapper, self).__init__()
        self.backbone = backbone
        self.model = self.init_backbone(checkpoint_path)

    def forward(self,
                x):
        with torch.no_grad():
            x = self.model(x)
        x = x.reshape(x.size(0), -1)
        return x

    def init_backbone(self,
                      checkpoint_path):
        """Helper to initialize a pretrained pytorch model.
        """
        if self.backbone == 'resnet50':
            model = resnet.resnet50(pretrained=False)  # Use Caffe ResNet instead
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet101':
            model = resnet.resnet101(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'resnet152':
            model = resnet.resnet152(pretrained=False)
            model.load_state_dict(torch.load(checkpoint_path))

        elif self.backbone == 'vgg16':
            model = models.vgg16(pretrained=True)

        elif self.backbone == 'vgg19':
            model = models.vgg19(pretrained=True)

        # Remove the last classifier layer
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)

        return model


# ----------------------------------------
# Functions for V2CNet
# ----------------------------------------

class Classification(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.conv1 = self._make_block(30, 2048)  # Output's shape: B, 16, 112, 112
        self.conv2 = self._make_block(2048, 1024)  # Output's shape: B, 32, 56, 56
        self.conv3 = self._make_block(1024, 512)  # Output's shape: B, 64, 28, 28

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512 * 256, out_features=256),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _make_block(self, in_channels, out_channels, kernel_size=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )


class VideoEncoder(nn.Module):
    """Module to encode pre-extracted features coming from
    pre-trained CNN.
    """

    def __init__(self,
                 in_size,
                 units):
        super(VideoEncoder, self).__init__()
        self.linear = nn.Linear(in_size, units)
        self.lstm = nn.LSTM(units, units, batch_first=True)
        self.reset_parameters()

    def forward(self,
                Xv):
        # Phase 1: Encoding Stage
        # Encode video features with one dense layer and lstm
        # State of this lstm to be used for lstm2 language generator
        Xv = self.linear(Xv)
        # print('linear:', Xv.shape)
        Xv = F.relu(Xv)

        Xv, (hi, ci) = self.lstm(Xv)
        Xv = Xv[:, -1, :]  # Only need the last timestep
        hi, ci = hi[0, :, :], ci[0, :, :]
        # print('lstm:', Xv.shape, 'hi:', hi.shape, 'ci:', ci.shape)
        return Xv, (hi, ci)

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


class CommandDecoder(nn.Module):
    """Module to decode features and generate word for captions
    using RNN.
    """

    def __init__(self,
                 units,
                 vocab_size,
                 embed_dim,
                 bias_vector=None):
        super(CommandDecoder, self).__init__()
        self.units = units
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, units)
        self.logits = nn.Linear(units, vocab_size, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.reset_parameters(bias_vector)

    def forward(self,
                Xs,
                states):
        # Phase 2: Decoding Stage
        # Given the previous word token, generate next caption word using lstm2
        # Sequence processing and generating
        # print('sentence decoding stage:')
        # print('Xs:', Xs.shape)
        Xs = self.embed(Xs)
        # print('embed:', Xs.shape)

        hi, ci = self.lstm_cell(Xs, states)
        # print('out:', hi.shape, 'hi:', states[0].shape, 'ci:', states[1].shape)

        x = self.logits(hi)
        # print('logits:', x.shape)
        x = self.softmax(x)
        # print('softmax:', x.shape)
        return x, (hi, ci)

    def init_hidden(self,
                    batch_size):
        """Initialize a zero state for LSTM.
        """
        h0 = torch.zeros(batch_size, self.units)
        c0 = torch.zeros(batch_size, self.units)
        return (h0, c0)

    def reset_parameters(self,
                         bias_vector):
        for n, p in self.named_parameters():
            if 'weight' in n:
                if 'hh' in n:
                    nn.init.orthogonal_(p.data)
                else:
                    nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        nn.init.uniform_(self.embed.weight.data, -0.05, 0.05)
        if bias_vector is not None:
            self.logits.bias.data = torch.from_numpy(bias_vector).float()


class CommandLoss(nn.Module):
    """Calculate Cross-entropy loss per word.
    """

    def __init__(self,
                 ignore_index=0):
        super(CommandLoss, self).__init__()
        self.cross_entropy = nn.NLLLoss(reduction='sum',
                                        ignore_index=ignore_index)

    def forward(self,
                input,
                target):
        return self.cross_entropy(input, target)


class Video2Command():
    """Train/Eval inference class for V2C model.
    """

    def __init__(self,
                 config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def build(self,
              bias_vector=None):

        # Initialize Encode & Decode models here
        self.video_encoder = VideoEncoder(in_size=list(self.config.BACKBONE.values())[0],
                                          units=self.config.UNITS)

        self.command_decoder = CommandDecoder(units=self.config.UNITS,
                                              vocab_size=self.config.VOCAB_SIZE,
                                              embed_dim=self.config.EMBED_SIZE,
                                              bias_vector=bias_vector)

        self.classification = Classification(num_classes=41)

        # Set model to gpu
        self.video_encoder.to(self.device)
        self.command_decoder.to(self.device)
        self.classification.to(self.device)

        # Loss function
        self.loss_objective = CommandLoss()
        self.loss_objective.to(self.device)

        # Setup parameters and optimizer
        self.params = list(self.video_encoder.parameters()) + \
                      list(self.command_decoder.parameters()) + \
                      list(self.classification.parameters())

        self.optimizer = torch.optim.Adam(self.params,
                                          lr=self.config.LEARNING_RATE)

        # Save configuration
        # Safely create checkpoint dir if non-exist
        if not os.path.exists(os.path.join(self.config.CHECKPOINT_PATH, 'saved')):
            os.makedirs(os.path.join(self.config.CHECKPOINT_PATH, 'saved'))

    def train(self,
              train_loader):
        """Train the Video2Command model.
        """

        def train_step(Xv, S, A):
            """One train step.
            """
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Action recognition
            action_preds = self.classification(Xv)
            action_loss = nn.CrossEntropyLoss()(action_preds, A)

            # Video feature extraction 1st
            Xv, states = self.video_encoder(Xv)
            # action = self.classification
            # Calculate mask against zero-padding
            S_mask = S != 0
            command_loss = 0.0
            # Teacher-Forcing for command decoder
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:, timestep]
                probs, states = self.command_decoder(Xs, states)
                # Calculate loss per word
                command_loss += self.loss_objective(probs, S[:, timestep + 1])
            command_loss = command_loss / S_mask.sum()  # Loss per word

            # Total loss
            loss = action_loss + command_loss

            # Gradient backward
            loss.backward()
            self.optimizer.step()
            return loss, action_loss, command_loss

            # return loss.item(), action_loss.item(), command_loss.item()

        # Training epochs
        self.video_encoder.train()
        self.command_decoder.train()
        self.classification.train()

        for epoch in range(self.config.NUM_EPOCHS):
            total_loss = 0.0
            total_action_loss = 0.0
            total_command_loss = 0.0
            for i, (Xv, S, A, clip_name) in enumerate(train_loader):
                # Mini-batch
                Xv, S, A = Xv.to(self.device), S.to(self.device), A.to(self.device)
                # Train step
                loss, action_loss, command_loss = train_step(Xv, S, A)
                total_loss += loss
                total_action_loss += action_loss
                total_command_loss += command_loss
                # Display
                if i % self.config.DISPLAY_EVERY == 0:
                    print('Epoch {}, Iter {}, Loss {:.6f}, '
                          'Action Loss {:.6f}, Command Loss {:.6f}'.format(
                                                            epoch + 1,
                                                                  i,
                                                                  loss,
                                                                  action_loss,
                                                                  command_loss))
            # End of epoch, save weights
            print('Total loss for epoch {}: {:.6f}'.format(epoch + 1, total_loss / (i + 1)))
            print('Total Action loss for epoch {}: {:.6f}'.format(epoch + 1, total_action_loss / (i + 1)))
            print('Total Command loss for epoch {}: {:.6f}'.format(epoch + 1, total_command_loss / (i + 1)))
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_weights(epoch + 1)
        return

    def evaluate(self,
                 test_loader,
                 vocab, practical=False):
        """
            Run the evaluation pipeline over the test dataset.
        """
        if not practical:
            assert self.config.MODE == 'test'
            y_pred, y_true, actions_pred, actions_true = [], [], [], []
            # Evaluation over the entire test dataset
            for i, (Xv, S_true, A_true, clip_names) in enumerate(test_loader):
                # Mini-batch
                Xv, S_true = Xv.to(self.device), S_true.to(self.device)
                S_pred, action = self.predict(Xv, vocab)
                y_pred.append(S_pred)
                y_true.append(S_true)
                actions_pred.append(action)
                actions_true.append(A_true)

            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            actions_pred = torch.cat(actions_pred, dim=0)
            actions_true = torch.cat(actions_true, dim=0)

            return y_pred.cpu().numpy(), y_true.cpu().numpy(), actions_pred.cpu().numpy(), actions_true.cpu().numpy()
        else:
            y_pred, actions_pred = [], []
            # Evaluation over the entire test dataset
            for i, (Xv, clip_names) in enumerate(test_loader):
                # Mini-batch
                Xv = Xv.to(self.device)
                S_pred, action = self.predict(Xv, vocab)
                y_pred.append(S_pred)
                actions_pred.append(action)
            actions_pred = torch.cat(actions_pred, dim=0)

            y_pred = torch.cat(y_pred, dim=0)

            return y_pred.cpu().numpy(), actions_pred.cpu().numpy()

    def predict(self,
                Xv,
                vocab):
        """Run the prediction pipeline given one sample.
        """
        self.video_encoder.eval()
        self.command_decoder.eval()
        self.classification.eval()

        with torch.no_grad():
            # Initialize S with '<sos>'
            S = torch.zeros((Xv.shape[0], self.config.MAXLEN), dtype=torch.long)
            S[:, 0] = vocab('<sos>')
            S = S.to(self.device)
            action = self.classification(Xv)
            # Start v2c prediction pipeline
            Xv, states = self.video_encoder(Xv)

            # states = self.command_decoder.reset_states(Xv.shape[0])
            # _, states = self.command_decoder(None, states, Xv=Xv)   # Encode video features 1st
            for timestep in range(self.config.MAXLEN - 1):
                Xs = S[:, timestep]
                probs, states = self.command_decoder(Xs, states)
                preds = torch.argmax(probs, dim=1)  # Collect prediction
                S[:, timestep + 1] = preds
        return S, action

    def save_weights(self,
                     epoch):
        """Save the current weights and record current training info
        into tensorboard.
        """
        # Save the current checkpoint
        torch.save({
            'VideoEncoder_state_dict': self.video_encoder.state_dict(),
            'CommandDecoder_state_dict': self.command_decoder.state_dict(),
            'Classification_state_dict': self.classification.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.CHECKPOINT_PATH, 'saved', 'v2c_epoch_{}.pth'.format(epoch)))
        print('Model saved.')

    def load_weights(self,
                     save_path):
        """Load pre-trained weights by path.
        """
        print('Loading...')
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
        self.video_encoder.load_state_dict(checkpoint['VideoEncoder_state_dict'])
        self.command_decoder.load_state_dict(checkpoint['CommandDecoder_state_dict'])
        self.classification.load_state_dict(checkpoint['Classification_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Model loaded.')