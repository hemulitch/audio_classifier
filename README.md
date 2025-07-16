# Audio Classification 

A PyTorch-based pipeline for UrbanSound audio classification with training,model evaluation, and FastAPI-based REST API for inference.

- **Models:** RNN trained on raw waveform, RNN trained on Log-Mel Spectrogram, and CNN trained on augmented data
- **REST API:** Simple file upload for prediction

## Getting Started

### Requirements
See [requirements.txt](./requirements.txt) for all dependencies.

### Installation

```bash
git clone https://github.com/hemulitch/audio_classifier.git
cd audio_classifier
pip install -r requirements.txt
```

### Usage

1. Train a Model
```
python main.py train --model cnn --epochs 20 --use_specaugment
```
Other models:
`--model rnn_raw`

`--model rnn_mel`

2. Evaluate a Model
   
```
python main.py eval --model cnn --weights cnn_weights.pt
```

3. Serve REST API
```
python main.py serve --model cnn --weights cnn_weights.pt
```

API will be available at:

http://localhost:8000/predict

### Data
Dataset can be found [here](https://urbansounddataset.weebly.com/urbansound8k.html)

UrbanSound8K data should be structured as:

/data/urbansound8k/

    data/
    
    train_part.csv
    
    val_part.csv
    

### Model Weights

After training, model weights are saved (e.g., `cnn_weights.pt`) and can be used for prediction or evaluation
