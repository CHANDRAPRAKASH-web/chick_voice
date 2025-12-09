
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class ChickenVoiceDataset(Dataset):
    """
    Loads chicken audio from:
        data/
          healthy/
          unhealthy/
          noise/
    and returns (mel_spectrogram, label).
    """

    def __init__(
        self,
        root_dir: str = "data",
        sample_rate: int = 16000,
        duration: float = 3.0,
        n_mels: int = 64,
    ):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels

        self.filepaths = []
        self.labels = []
        self.label2idx = {}

        # ---- scan folders and collect file paths ----
        classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

        for idx, cls in enumerate(classes):
            self.label2idx[cls] = idx
            folder = os.path.join(root_dir, cls)
            for fname in os.listdir(folder):
                if fname.lower().endswith(".wav"):
                    self.filepaths.append(os.path.join(folder, fname))
                    self.labels.append(idx)

        print("Classes found:", self.label2idx)
        print("Total audio files:", len(self.filepaths))

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        # ---- load audio ----
        audio, sr = librosa.load(path, sr=self.sample_rate)

        # fixed length
        target_len = int(self.sample_rate * self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # ---- mel spectrogram ----
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        # normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # [1, n_mels, time]
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor


if __name__ == "__main__":
    print("Testing ChickenVoiceDataset...")
    ds = ChickenVoiceDataset("data")
    print("Length:", len(ds))
    if len(ds) > 0:
        x, y = ds[0]
        print("Sample shape:", x.shape)
        print("Label index:", y.item())