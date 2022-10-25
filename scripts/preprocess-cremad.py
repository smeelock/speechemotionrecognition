"""
A script to preprocess the CREMA-D dataset.
"""

import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile

data_dir = Path('../data/')
raw_dataset_dir = data_dir / 'raw' / 'CREMA-D'
raw_audio_dir = raw_dataset_dir / 'AudioWAV/'
preprocessed_audio_dir = data_dir / 'preprocessed/CREMA-D/AudioWAV'
labels_filepath = raw_dataset_dir / 'processedResults/summaryTable.csv'


def get_cremad_filepaths_and_labels():
    df = pd.read_csv(labels_filepath, header=0, index_col=False, usecols=['FileName', 'VoiceVote'])
    df.rename(columns={'FileName': 'filename', 'VoiceVote': 'label'}, inplace=True)

    nonunique_emotion_filter = df['label'].str.contains(':', regex=True)
    df = df[~nonunique_emotion_filter]

    return df['filename'].to_list(), df['label'].to_list()


def preprocess_cremad(filenames):
    board = Pedalboard([])  # create pedalboard pipeline

    for filename in tqdm(filenames, desc='Preprocessing CREMA-D dataset'):
        input_filepath = os.path.join(raw_audio_dir, filename + '.wav')
        with AudioFile(input_filepath) as f:
            audio = f.read(f.frames)
            samplerate = f.samplerate

        effected = board(audio, samplerate)  # pass audio through pedalboard pipeline

        output_filepath = os.path.join(preprocessed_audio_dir, filename + '.wav')
        with AudioFile(output_filepath, 'w', samplerate, effected.shape[0]) as f:
            f.write(effected)


# ===== Main =====
if __name__ == "__main__":
    # Set working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    # Get CREMA-D file paths and labels
    filenames, labels = get_cremad_filepaths_and_labels()

    # Preprocess dataset
    preprocess_cremad(filenames)
