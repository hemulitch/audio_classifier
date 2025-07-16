import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
torchaudio.set_audio_backend("soundfile")
from torchaudio.transforms import MelSpectrogram
from fastapi import FastAPI, UploadFile, File
import uvicorn
import random

# Classes and constants
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


############################
#      DATASET CLASS       #
############################
class AudioDataset(Dataset):
    def __init__(self, path_to_csv, path_to_folder, pad_size=384000, sr=44100):
        self.csv = pd.read_csv(path_to_csv)[["ID", "Class"]]
        self.path_to_folder = path_to_folder
        self.pad_size = pad_size
        self.sr = sr
        self.class_to_idx = {CLASSES[i]: i for i in range(10)}

    def __getitem__(self, index):
        output = self.csv.iloc[index]
        path = os.path.join(self.path_to_folder, str(output['ID']) + '.wav')
        y = self.class_to_idx[output['Class']]

        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(wav)

        wav = wav[0]
        length = wav.shape[0]
        if wav.shape[0] < self.pad_size:
            wav = torch.nn.functional.pad(wav, (0, self.pad_size - wav.shape[0]))
        elif wav.shape[0] > self.pad_size:
            wav = wav[:self.pad_size]

        return {'x': wav, 'y': y, 'len': length}

    def __len__(self):
        return self.csv.shape[0]


############################
#       MODELS             #
############################
class RecurrentRawAudioClassifier(nn.Module):
    def __init__(self, num_classes=10, window_length=1024, hop_length=256, hidden=256, num_layers=2):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length
        self.first_mlp = nn.Sequential(
            nn.Linear(self.window_length, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU()
        )
        self.rnn = nn.LSTM(
            input_size=16, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden * num_layers, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lens):
        batch_windows = x.unfold(1, self.window_length, self.hop_length)
        batch_windows_features = self.first_mlp(batch_windows)
        _, (hidden_states, _) = self.rnn(batch_windows_features)
        # (num_layers, num_directions, batch, hidden_size)
        batch = x.shape[0]
        hidden_states = hidden_states.view(self.rnn.num_layers, 2, batch, self.rnn.hidden_size)
        hidden_flattened = hidden_states.permute(2, 0, 1, 3).reshape(batch, -1)
        logits = self.final_mlp(hidden_flattened)
        return logits


class RecurrentMelSpectClassifier(nn.Module):
    def __init__(self, num_classes=10, hidden=256, num_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=64, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(2 * hidden * num_layers, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lens):
        x = x.transpose(1, 2)  # (batch, time, mel)
        _, (hidden_states, _) = self.rnn(x)
        batch = x.shape[0]
        hidden_states = hidden_states.view(self.rnn.num_layers, 2, batch, self.rnn.hidden_size)
        hidden_flattened = hidden_states.permute(2, 0, 1, 3).reshape(batch, -1)
        logits = self.final_mlp(hidden_flattened)
        return logits


class CNN10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, lens):
        z = self.cnn_backbone(x[:, None, :, :])
        z = torch.nn.functional.max_pool2d(z, kernel_size=z.size()[2:])[:, :, 0, 0]
        return self.final_mlp(z)


############################
#      AUGMENTATION        #
############################
class SpecAugment:
    def __init__(
        self, filling_value="mean", n_freq_masks=2, n_time_masks=2,
        max_freq=10, max_time=50
    ):
        self.filling_value = filling_value
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_freq = max_freq
        self.max_time = max_time

    def _get_mask_value(self, spect):
        if self.filling_value == 'mean':
            return spect.mean()
        elif self.filling_value == 'min':
            return spect.min()
        elif self.filling_value == 'max':
            return spect.max()
        else:
            return float(self.filling_value)

    def __call__(self, spect, lens):
        # spect: [B, 64, T], lens: not used
        mask_value = self._get_mask_value(spect)
        B, F, T = spect.shape
        for b in range(B):
            for _ in range(self.n_time_masks):
                t = random.randint(0, max(0, T - 1 - self.max_time))
                w = random.randint(1, self.max_time)
                spect[b, :, t:t + w] = mask_value
            for _ in range(self.n_freq_masks):
                f = random.randint(0, max(0, F - 1 - self.max_freq))
                w = random.randint(1, self.max_freq)
                spect[b, f:f + w, :] = mask_value
        return spect, lens

############################
#    FEATURE EXTRACTORS    #
############################
def compute_log_melspectrogram(wav_batch, lens, sr, device="cpu"):
    featurizer = MelSpectrogram(
        sample_rate=sr, n_fft=1024, win_length=1024, hop_length=256,
        n_mels=64, center=False,
    ).to(device)
    return torch.log(featurizer(wav_batch).clamp(1e-5)), lens // 256


############################
#      TRAIN/VAL UTILS     #
############################
def train_audio_clfr(
    model, optimizer, train_dataloader, sr,
    criterion=nn.CrossEntropyLoss(),
    data_transform=None, augmentation=None,
    num_epochs=10, device=DEVICE, verbose_num_iters=100
):
    model.to(device)
    model.train()
    iter_i = 0
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            lens = batch["len"].to(device)
            if data_transform:
                x, lens = data_transform(x, lens, device=device, sr=sr)
            if augmentation:
                x, lens = augmentation(x, lens)
            probs = model(x, lens)
            optimizer.zero_grad()
            loss = criterion(probs, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pred_cls = probs.argmax(dim=-1)
            train_accuracies.append((pred_cls == y).float().mean().item())
            iter_i += 1
            if iter_i % verbose_num_iters == 0:
                print(f"Epoch {epoch}, iter {iter_i}, loss: {loss.item():.4f}, acc: {train_accuracies[-1]:.3f}")
    model.eval()


def accuracy(model, val_dataloader, sr, device, data_transform=None):
    pred_true_pairs = []
    model.eval()
    for batch in val_dataloader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lens = batch["len"].to(device)
        with torch.no_grad():
            if data_transform:
                x, lens = data_transform(x, lens, sr=sr, device=device)
            probs = model(x, lens)
            pred_cls = probs.argmax(dim=-1)
        for pred, true in zip(pred_cls.cpu().numpy(), y.cpu().numpy()):
            pred_true_pairs.append((pred, true))
    acc = np.mean([p[0] == p[1] for p in pred_true_pairs])
    print(f"Val accuracy: {acc:.3f}")
    return acc

############################
#         CLI MAIN         #
############################
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # Training
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--model', choices=['rnn_raw', 'rnn_mel', 'cnn'], required=True)
    train_parser.add_argument('--epochs', type=int, default=10)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--data_dir', default='./data/urbansound8k/data')
    train_parser.add_argument('--train_csv', default='./data/urbansound8k/train_part.csv')
    train_parser.add_argument('--val_csv', default='./data/urbansound8k/val_part.csv')
    train_parser.add_argument('--use_specaugment', action='store_true')

    # Evaluation
    eval_parser = subparsers.add_parser('eval')
    eval_parser.add_argument('--model', choices=['rnn_raw', 'rnn_mel', 'cnn'], required=True)
    eval_parser.add_argument('--weights', type=str, required=True)
    eval_parser.add_argument('--data_dir', default='./data/urbansound8k/data')
    eval_parser.add_argument('--val_csv', default='./data/urbansound8k/val_part.csv')

    # REST API
    serve_parser = subparsers.add_parser('serve')
    serve_parser.add_argument('--model', choices=['rnn_raw', 'rnn_mel', 'cnn'], required=True)
    serve_parser.add_argument('--weights', type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        # Data
        train_dataset = AudioDataset(args.train_csv, args.data_dir)
        val_dataset = AudioDataset(args.val_csv, args.data_dir)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, drop_last=True)

        # Model
        if args.model == "rnn_raw":
            model = RecurrentRawAudioClassifier()
            transform = None
            aug = None
        elif args.model == "rnn_mel":
            model = RecurrentMelSpectClassifier()
            transform = compute_log_melspectrogram
            aug = None
        elif args.model == "cnn":
            model = CNN10()
            transform = compute_log_melspectrogram
            aug = SpecAugment() if args.use_specaugment else None

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_audio_clfr(
            model, optimizer, train_loader, train_dataset.sr,
            data_transform=transform, augmentation=aug, num_epochs=args.epochs
        )
        # Save weights
        torch.save(model.state_dict(), f"{args.model}_weights.pt")
        print(f"Model saved as {args.model}_weights.pt")
        # Eval
        accuracy(model, val_loader, train_dataset.sr, DEVICE, data_transform=transform)

    elif args.command == "eval":
        val_dataset = AudioDataset(args.val_csv, args.data_dir)
        val_loader = DataLoader(val_dataset, batch_size=32, pin_memory=True, drop_last=True)
        if args.model == "rnn_raw":
            model = RecurrentRawAudioClassifier()
            transform = None
        elif args.model == "rnn_mel":
            model = RecurrentMelSpectClassifier()
            transform = compute_log_melspectrogram
        elif args.model == "cnn":
            model = CNN10()
            transform = compute_log_melspectrogram
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
        model.to(DEVICE)
        accuracy(model, val_loader, val_dataset.sr, DEVICE, data_transform=transform)

    elif args.command == "serve":
        # FastAPI REST service
        model_name = args.model
        weights_path = args.weights
        if model_name == "rnn_raw":
            model = RecurrentRawAudioClassifier()
            transform = None
        elif model_name == "rnn_mel":
            model = RecurrentMelSpectClassifier()
            transform = compute_log_melspectrogram
        elif model_name == "cnn":
            model = CNN10()
            transform = compute_log_melspectrogram
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        app = FastAPI()

        @app.post("/predict")
        async def predict(file: UploadFile = File(...)):
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            wav, sr = torchaudio.load(tmp_path)
            wav = wav[0]
            # Pad/crop to 384000
            pad_size = 384000
            if wav.shape[0] < pad_size:
                wav = torch.nn.functional.pad(wav, (0, pad_size - wav.shape[0]))
            elif wav.shape[0] > pad_size:
                wav = wav[:pad_size]
            x = wav.unsqueeze(0).to(DEVICE)
            lens = torch.tensor([wav.shape[0]])
            if transform:
                x, lens = transform(x, lens, sr=44100, device=DEVICE)
            with torch.no_grad():
                out = model(x, lens)
                pred = int(out.argmax(-1).cpu().numpy()[0])
                pred_label = CLASSES[pred]
            os.remove(tmp_path)
            return {"class": pred_label, "index": pred}

        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
