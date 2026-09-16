# Food-11 Image Classification with MLflow

This project trains a food image classifier using the Food-11 dataset. It uses a pretrained ResNet-18 model from PyTorch and MLflow to track training experiments, hyperparameters, metrics, and saved models.

## Features

- Training a pretrained ResNet-18 model for 11 food classes
- Using the processed Food-11 image dataset
- Training on a mini dataset for faster experimentation
- Logging parameters, metrics, and trained models with MLflow
- Comparing multiple experiments in the MLflow UI
- Managing Python dependencies with `uv`

## Project Structure

```text
mlops-lab-1/
├── data/
│   ├── food11_processed/
│   └── food11_processed_mini/
├── src/
│   └── food11/
│       └── train.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

Install the project dependencies with:

```bash
uv sync
```

The project uses:

```text
mlflow
torch
torchvision
scikit-learn
pillow
```

## Start MLflow

Start the MLflow tracking server from the root of the repository:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Open the MLflow UI in a browser:

```text
http://127.0.0.1:5000
```

MLflow stores:

- Run metadata, parameters, metrics, timestamps, and statuses in `mlflow.db`
- Model artifacts and other logged files in `mlruns/`

These files are local generated outputs and should not be committed to Git.

## Train the Model

Run the model-training script using the mini dataset:

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

The training script accepts the following arguments:

| Argument | Description |
|---|---|
| `--dataset` | Dataset to use: `mini` or `processed` |
| `--epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `--batch-size` | Number of images per training batch |

Example:

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
```

## MLflow Logging

Each training run is logged in the MLflow experiment named `food11`.

The following parameters are logged:

```text
dataset
epochs
lr
batch_size
model
num_classes
image_size
```

The following metrics are logged:

```text
train_loss
val_loss
val_accuracy
test_accuracy
```

The trained PyTorch model is also logged as an MLflow artifact named `model`.

## Experiment Results

Four training experiments were run using the mini Food-11 dataset for 5 epochs.

| Learning rate | Batch size | Validation accuracy | Test accuracy |
|---:|---:|---:|---:|
| 0.01 | 32 | 0.161 | 0.156 |
| 0.001 | 32 | 0.513 | 0.575 |
| 0.0001 | 32 | **0.707** | **0.739** |
| 0.001 | 64 | 0.494 | 0.527 |

The best run used a learning rate of `0.0001` and batch size of `32`.

```text
Run name: victorious-wren-333
Run ID: b0895843421943b88497c9585158c840
Validation accuracy: 0.707
Test accuracy: 0.739
```

The experiment showed that a high learning rate of `0.01` performed poorly. For this ResNet-18 fine-tuning task, the lower learning rate of `0.0001` produced the best validation and test accuracy.

```
## Docker Environment

MLflow was run inside a local Docker container because Windows Code Integrity blocked the native pandas extension required by MLflow in the Windows Python environment. Docker provided an isolated Linux environment where MLflow could run correctly. The container exposed port `5000`, allowing the MLflow UI to be accessed from the browser at `http://127.0.0.1:5000`. Docker was used only as a local execution workaround; no Docker image or container is included in this repository.