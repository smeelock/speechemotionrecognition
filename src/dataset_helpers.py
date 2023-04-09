import os
import torch
from datasets import Dataset, Audio, Value
from pathlib import Path

from constants import DEFAULT_TARGET_SAMPLING_RATE, DEFAULT_IEMOCAP_LABEL_LIST

def _get_wavs_paths(data_dir):
    wav_dir = data_dir / "sentences" / "wav"
    wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths

def _create_metadata_csv(root_dir):
    root = Path(root)
    self._path = root / "IEMOCAP"
    assert os.path.isdir(self._path), "Dataset not found."

    sessions = (1,2,3,4,5)

    all_data = []
    data = []
    mapping = []

    for session in sessions:
        session_name = f"Session{session}"
        session_dir = self._path / session_name

        # get wav paths
        wav_paths = _get_wavs_paths(session_dir)
        for wav_path in wav_paths:
            wav_stem = str(Path(wav_path).stem)
            all_data.append(wav_stem)

        # add labels
        label_dir = session_dir / "dialog" / "EmoEvaluation"
        query = "*.txt"
        if utterance_type == "scripted":
            query = "*script*.txt"
        elif utterance_type == "improvised":
            query = "*impro*.txt"
        label_paths = label_dir.glob(query)

        for label_path in label_paths:
            with open(label_path, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    line = re.split("[\t\n]", line)
                    wav_stem = line[1]
                    label = line[2]
                    if wav_stem not in all_data:
                        continue
                    if label not in DEFAULT_IEMOCAP_LABEL_LIST:
                        continue
                    mapping[wav_stem] = {}
                    mapping[wav_stem]["label"] = label

        for wav_path in wav_paths:
            wav_stem = str(Path(wav_path).stem)
            if wav_stem in mapping:
                data.append(wav_stem)
                mapping[wav_stem]["path"] = wav_path

features = Dataset.Features({
    "wav": "",
    "path": Value(dtype='string', id=None),
    "sample_rate": Value(dtype='int', id=None),
    "label": Value(dtype='string', id=None),
    "speaker": Value(dtype='string', id=None)
})
iemocap = Dataset.from_dict("audiofolder", data_files=all_wav_paths)

class ProcessedIEMOCAP(torch.utils.data.Dataset):
    def __init__(self, data: IEMOCAP, processor, config):
        self.data = data
        self.processor = processor
        self.config = config

        if hasattr(config, "target_sampling_rate"):
            self.target_sampling_rate = config.target_sampling_rate
        elif (
            hasattr(self.processor, "feature_extractor")
            and hasattr(self.processor.feature_extractor, "sampling_rate")
        ):
            self.target_sampling_rate = self.processor.feature_extractor.sampling_rate
        else:
            self.target_sampling_rate = DEFAULT_TARGET_SAMPLING_RATE

    def __getitem__(self, index):
        wav, _, _, label, _ = self.data[index]
        inputs = self.processor(wav.squeeze(), sampling_rate=self.target_sampling_rate)
        inputs["labels"] = self.config.label2id[label]
        return inputs

    def __len__(self):
        return len(self.data)
