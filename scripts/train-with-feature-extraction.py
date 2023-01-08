"""
A script to use WavLM as a feature extractor and train a model to recognise emotions on the CREMA-D dataset.
"""

import argparse
import copy
import datetime
import os
import time

import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WavLMModel

# ========= Configuration =========
MODELS_DIR = '../models'

# parameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9

EPOCHS = 25
BATCH_SIZE = 1

SCALING_FACTOR = 1e-4
MAX_MAGNITUDE = 10

N_OUT_FEATURES = 256
N_HIDDEN_STATE = 768
N_CLASSES = 6  # ang, dis, fea, hap, neu, sad

# arg parser
parser = argparse.ArgumentParser(description='A script to use WavLM as a feature extractor and train a model to '
                                             'recognise emotions on the CREMA-D dataset.')


# ========= Helper functions =========
def get_dataloaders():
    # load dataset using tensorflow_datasets
    dataset, info = tfds.load('crema_d', split=['train', 'validation'], shuffle_files=True, batch_size=BATCH_SIZE,
                              as_supervised=True, with_info=True, data_dir='../data/raw')

    dataloaders = {
        'train': dataset[0],
        'val': dataset[1],
    }
    sampling_rate = info.features["audio"].sample_rate
    return dataloaders, sampling_rate


def get_model_and_processor():
    class FEWavLMModel(nn.Module):
        def __init__(self):
            super(FEWavLMModel, self).__init__()
            # download pretrained model
            self.wavlm = WavLMModel.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

            # freeze feature-extraction layers
            for param in self.wavlm.parameters():
                param.requires_grad = False

            self.head = nn.Linear(N_OUT_FEATURES * N_HIDDEN_STATE, 9)

        def forward(self, **inputs):
            x = self.wavlm(**inputs).last_hidden_state

            _, max_length, n_hidden_units = x.shape

            x = F.pad(x, (0, 0, 0, N_OUT_FEATURES - max_length), 'constant', 0)  # right side zero padding
            x = torch.flatten(x, 1, -1)
            x = self.head(x)
            return x

    model = FEWavLMModel()
    processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    return model, processor


def train(model, dataloaders, criterion, optimizer, num_epochs, device, model_assets={}, dataset_assets={}):
    # check inputs
    assert 'processor' in model_assets, "`processor` is not given"
    processor = model_assets['processor']
    assert 'sampling_rate' in dataset_assets, "`sampling_rate` is not given"
    sampling_rate = dataset_assets['sampling_rate']

    # init
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # training loop
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            length = 1
            for inputs, labels in tqdm(dataloaders[phase].as_numpy_iterator(), desc=phase):  # audio, label, speaker_id
                length += 1
                inputs = inputs.squeeze()
                inputs = torch.from_numpy(inputs.copy())
                labels = torch.from_numpy(labels)
                inputs = torch.mul(inputs, SCALING_FACTOR)  # scale
                inputs = processor(inputs, sampling_rate=sampling_rate, return_tensors='pt')
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(**inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs['input_values'].size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / length
            epoch_acc = running_corrects / length

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc


# ========= Main =========
if __name__ == "__main__":
    args = parser.parse_args()

    # Set working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    # Get dataloaders
    dataloaders, sampling_rate = get_dataloaders()

    # Get model
    model, processor = get_model_and_processor()

    # Initialize training
    print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)

    optimizer_ft = optim.SGD(params_to_update, lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Train model
    model = model.to(device)
    model_ft, hist, best_acc = train(
        model,
        dataloaders,
        criterion,
        optimizer_ft,
        num_epochs=EPOCHS,
        device=device,
        model_assets={'processor': processor},
        dataset_assets={'sampling_rate': sampling_rate}
    )

    # Save model
    model_name = "{date}-{user}-{model}-{experiment}-val_acc_is_{val_acc:.4f}".format(
        date=datetime.date.today().strftime('%Y-%m-%d'),
        user="avn",
        model="wavlm",
        experiment="fe",  # fe=feature extraction
        val_acc=best_acc
    ).replace('.', '_')
    torch.save(model_ft, f"{MODELS_DIR}/{model_name}.pt")
