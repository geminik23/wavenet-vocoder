import librosa
import numpy as np
import os
from glob import glob
from utils import mulaw_encode
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
import torch
import random
from config import Config

class MulawMelDataset(Dataset):
    def __init__(self, files, audio_length, config):
        self.audio_length = audio_length
        self.config = config
        self.files = files
        
    
    def _compute_mel_spec(self, y):
        return librosa.feature.melspectrogram(y=y,
                                              n_mels=self.config.n_mels,
                                              sr=self.config.sample_rate, 
                                              n_fft=self.config.fft_size, 
                                              hop_length=self.config.hop_size,
                                              win_length=self.config.win_size,
                                              power=self.config.power)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        y, sr = librosa.load(self.files[index])
        if sr != self.config.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.config.sample_rate)

        max_v = np.abs(y).max()
        if max_v >= 1.0:
            y = y*(0.998/max_v)

        # two cases 
        if len(y) < self.audio_length:
            y = np.pad(y, (0, self.audio_length - len(y)))
        else:
            bidx = random.randint(0, len(y) - self.audio_length)
            y = y[bidx:bidx+self.audio_length]
        
        # encode
        melspecs = torch.tensor(self._compute_mel_spec(y))
        y = mulaw_encode(y, self.config.num_class-1)
        y = torch.tensor(y).long()
        return (y, melspecs), y



def load_ljspeech_dataset(config):
    """
    Returns:
        tuple: (train dataset, test dataset)
    """
    files = glob(os.path.join(config.dataset_path, f"{config.lj_folder_name}/wavs/*.wav"))
    test_len = int(len(files)/20)
    # train, test = random_split(files, [len(files)-test_len, test_len])
    train, test = files[:len(files)-test_len], files[-test_len:]
    return MulawMelDataset(train, config.lj_train_audio_length, config), MulawMelDataset(test, config.lj_train_audio_length, config)


def _load_list(path, filename, filter):
    filepath = os.path.join(path, filename)
    with open(filepath) as f:
        return [os.path.normpath(os.path.join(path, line.strip())) for line in f if line.split('/')[0] in filter]

def load_speechcommand_dataset(config:Config):
    """
    Returns:
        tuple: (train dataset, test dataset)
    """
    LABELS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    d = SPEECHCOMMANDS(config.dataset_path, download=True)
    test_files = _load_list(d._path, "validation_list.txt", LABELS) + _load_list(d._path, "testing_list.txt", LABELS)
    e = set(test_files)
    files = []
    for l in LABELS:
        files += glob(os.path.join(d._path, l) + "/*.wav")
    train_files = [f for f in files if f not in e]
    return MulawMelDataset(train_files, config.sc_train_audio_length, config), MulawMelDataset(test_files, config.sc_train_audio_length, config)
    



