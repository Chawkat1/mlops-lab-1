# Food-11 Model Serving with Docker and Docker Compose

This picks up where `READMELAB2.md` left off. Lab 2 produced a trained ResNet-18 model tracked with MLflow. This part registers that model in the MLflow Model Registry, serves it behind a FastAPI API, containerizes everything with Docker, and finally orchestrates a full `mlflow` + `inference` + `frontend` stack with Docker Compose.

## Features

- Registering a trained model in the MLflow Model Registry under a shared name (`food11`)
- Assigning the `champion` alias to the version currently in service
- Serving predictions with FastAPI, loading the model via `models:/food11@champion` instead of a local file
- Packaging the API into a Docker image with a multi-stage, CPU-only build
- Orchestrating mlflow, the inference API, and a Streamlit frontend together with Docker Compose
- A named Docker volume so the MLflow tracking database and artifacts survive `docker compose down` / `up`

## Project Structure

```text
mlops-lab-1/
├── Dockerfile                 # Lab 3: multi-stage build for the inference API
├── .dockerignore
├── docker-compose.yml         # Lab 4: wires mlflow + inference + frontend together
├── mlflow/
│   └── Dockerfile             # Lab 4: containerized mlflow tracking server
├── frontend/
│   ├── Dockerfile             # Lab 4: Streamlit upload page
│   └── app.py
├── src/
│   └── food11/
│       ├── train.py           # from Lab 2
│       └── serve.py           # Lab 3: FastAPI serving app
├── lab/
│   ├── lab3-docker.md         # Lab 3 instructions + answers
│   └── lab4-compose.md        # Lab 4 instructions + answers
└── ...
```

## Lab 3 — Registering the model and serving it with Docker

### Register the best run and set the champion alias

The best run from Lab 2 was promoted to the Model Registry under the name `food11`, and the `champion` alias was pointed at that version:

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.register_model('runs:/<run-id>/model', 'food11')
"

uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('food11', 'champion', <version-number>)
"
```

Because the alias is a mutable pointer, promoting a new model version later only means moving `champion`, the serving code never has to change.

### Serve the model with FastAPI

`src/food11/serve.py` exposes:

- `GET /health` — returns `{"status": "ok"}`
- `POST /predict` — accepts an uploaded image and returns the predicted category and confidence score

It loads the model once at startup with `mlflow.pyfunc.load_model("models:/food11@champion")` and reads the tracking server address from the `MLFLOW_TRACKING_URI` environment variable (defaulting to `http://127.0.0.1:5000`), so the same code works locally and inside a container.

Run it locally:

```bash
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

### Build and run the Docker image

The `Dockerfile` at the repo root is a two-stage build: a `builder` stage installs dependencies with `uv sync` and pulls CPU-only `torch`/`torchvision` wheels from `download.pytorch.org/whl/cpu` (avoiding the multi-GB CUDA build), and a slim `runtime` stage copies over only the finished virtual environment and `src/`.

```bash
docker build -t food11-api:latest .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
```

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_0.jpg" http://127.0.0.1:8000/predict
```

Full write-up and answers to all 9 lab questions: [`lab/lab3-docker.md`](lab/lab3-docker.md).

## Lab 4 — Orchestrating mlflow, inference, and a frontend with Docker Compose

### Services

`docker-compose.yml` defines three services on one private network:

| Service     | Built from       | Published port | Purpose                                    |
|-------------|-------------------|-----------------|---------------------------------------------|
| `mlflow`    | `./mlflow`         | `5000`          | Tracking server + model registry            |
| `inference` | `.` (Lab 3 image)  | *(none)*        | FastAPI serving API, reached only by `frontend` |
| `frontend`  | `./frontend`       | `8501`          | Streamlit upload page                       |

`inference` doesn't publish a port because nothing outside the Compose network ever calls it directly; it's reached at `http://inference:8000` over Compose's own DNS. The mlflow database and artifacts live on a named volume (`mlflow-data`) so they survive `docker compose down` / `up` (but not `down -v`, which deletes the volume too).

> **Note on the mlflow image:** this project's mlflow version (3.16.x) requires two flags beyond a bare-bones server command: `--allowed-hosts` (mlflow now rejects requests whose `Host` header — like the Compose service name `mlflow` — isn't allowlisted) and `--artifacts-destination` in place of a literal `--default-artifact-root` path (otherwise the registry records a raw filesystem path that only the mlflow container itself can read, and `inference` can never download the model). Both are set in `mlflow/Dockerfile`.

### Bring the stack up

```bash
docker compose up --build
```

Then, since this mlflow instance starts empty, train and register a model into it (same commands as Lab 2/3, now pointed at the containerized mlflow):

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32

uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
result = mlflow.register_model('runs:/<run-id>/model', 'food11')
mlflow.MlflowClient().set_registered_model_alias('food11', 'champion', result.version)
"

docker compose restart inference
```

Open the mlflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) and the frontend at [http://127.0.0.1:8501](http://127.0.0.1:8501). Upload a food image and it returns a prediction.

### Verified in this session

- `food11` registered with a `champion` alias, currently version 1 (val_accuracy 0.726, test_accuracy 0.746 on the mini dataset).
- `inference` loads the champion model at startup and serves correct predictions over the internal Compose network.
- Persistence: `docker compose down` + `up` keeps the registry (named volume untouched); `docker compose down -v` + `up` wipes it completely, requiring the model to be retrained and re-registered.
- `depends_on` only controls start order, not readiness — restarting `inference` before mlflow has a registered model crashes it with `RESOURCE_DOES_NOT_EXIST`, confirmed directly.

Full write-up and answers to all 11 lab questions: [`lab/lab4-compose.md`](lab/lab4-compose.md).
