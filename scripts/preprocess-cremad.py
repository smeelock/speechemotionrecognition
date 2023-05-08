"""
A script to preprocess the CREMA-D dataset.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from pedalboard import Pedalboard
from pedalboard.io import AudioFile
from tqdm import tqdm

# ========= Configuration =========
DEFAULT_SOURCE_DIR = '../data/raw/CREMA-D'
DEFAULT_PREPROCESSED_AUDIO_DIR = '../data/preprocessed/CREMA-D/AudioWAV'

# arg parser
parser = argparse.ArgumentParser(description='A script to preprocess the CREMA-D dataset.')
parser.add_argument('--source', default=DEFAULT_SOURCE_DIR, dest='source',
                    help=f'Path to dataset (default is "{DEFAULT_SOURCE_DIR}"). Root directory of dataset where CREMA-D/AudioWAV/ and CREMA-D/processedResults/summaryTable.csv exist.')
parser.add_argument('--destination', default=DEFAULT_PREPROCESSED_AUDIO_DIR, dest='destination',
                    help=f'Path to save preprocessed dataset (default is "{DEFAULT_PREPROCESSED_AUDIO_DIR}")')


# ========= Helper functions =========
def get_cremad_filepaths_and_labels(source_dir=DEFAULT_SOURCE_DIR):
    source_dir = Path(source_dir)
    labels_filepath = source_dir / 'processedResults/summaryTable.csv'

    df = pd.read_csv(labels_filepath, header=0, index_col=False, usecols=['FileName', 'VoiceVote'])
    df.rename(columns={'FileName': 'filename', 'VoiceVote': 'label'}, inplace=True)

    nonunique_emotion_filter = df['label'].str.contains(':', regex=True)
    df = df[~nonunique_emotion_filter]

    return df['filename'].to_list(), df['label'].to_list()


def preprocess_cremad(filenames, source_dir=DEFAULT_SOURCE_DIR, destination_dir=DEFAULT_PREPROCESSED_AUDIO_DIR):
    raw_audio_dir = Path(source_dir) / 'AudioWAV'
    preprocessed_audio_dir = Path(destination_dir)

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


# ========= Main =========
if __name__ == "__main__":
    args = parser.parse_args()

    # Set working directory
    abspath = os.path.abspath(__file__)
    dirname = os.path.dirname(abspath)
    os.chdir(dirname)

    # Get CREMA-D file paths and labels
    filenames, labels = get_cremad_filepaths_and_labels(source_dir=args.source)

    # Preprocess dataset
    preprocess_cremad(filenames, source_dir=args.source, destination_dir=args.destination)
