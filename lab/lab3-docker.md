# Lab 3 - Containerizing the model with Docker

This lab continues the project from Lab 1 (git+dvc data) and Lab 2 (training + mlflow tracking). You now have a trained model and a run ID for your best experiment. In this lab you will register that model in the mlflow Model Registry, write a small serving API around it, and package everything into a Docker image so it can run anywhere without a local Python setup.

> What you need to know:
> - a Docker **image** is a built, immutable set of layers (filesystem + metadata); a **container** is a running instance of an image
> - a `Dockerfile` describes how to build an image, one instruction (layer) at a time; Docker caches layers, so ordering instructions from least-to-most-frequently-changed speeds up rebuilds
> - a **multi-stage build** uses one stage to install/build dependencies and a second, smaller stage to run the app, so build tools don't bloat the final image
> - the mlflow **Model Registry** lets you name and version a model (e.g. `food11`) independently of the run that produced it; instead of the old (deprecated) stages like `Staging`/`Production`, current mlflow uses **aliases** — mutable pointers like `champion` that you can reassign to any version — plus free-form **tags** for anything else you want to track
> - a container is isolated from the host network by default; reaching a service running on your host machine (like the mlflow tracking server from Lab 2) requires extra configuration

## Before You Start: What You Need Installed

- **Docker** — Docker Desktop (Mac/Windows) or Docker Engine (Linux). Confirm with `docker --version` and `docker run hello-world`.
- Your Lab 2 project, with at least one completed training run and its run ID noted.

## Register your best model

### Promote a run to the Model Registry

Using the run ID you noted at the end of Lab 2 (the run with the best `val_accuracy`), register its model artifact under a shared name. You can do this either from the mlflow UI or from code — pick whichever you prefer:

**Option A — from the UI:**

1. Open the `food11` experiment, sort the runs table by `val_accuracy` descending, and open your best run.
2. In the run page, open the "Artifacts" tab, select the `model` folder, and click "Register Model".
3. Choose "Create New Model", name it `food11`, and confirm.

**Option B — from code:**

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.register_model('runs:/<your-run-id>/model', 'food11')
"
```

> Question 1: Open the "Models" tab in the mlflow UI. What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

### Assign it the `champion` alias

Model Registry stages (`Staging`/`Production`) are deprecated in current mlflow versions. The replacement is **aliases**: named, mutable pointers to a specific model version (e.g. `champion`, `challenger`) that you can reassign at any time without changing the version number itself.

**Option A — from the UI:**

In the mlflow UI, open the `food11` registered model, select the version you just created, and add the alias `champion` to it.

**Option B — from code:**

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('food11', 'champion', <your-version-number>)
"
```

> Question 2: What aliases replaced the old built-in stages in mlflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

## Write a serving script

Create `./src/food11/serve.py`. It should:

1. Use **FastAPI** to expose:
   - `GET /health` — returns `{"status": "ok"}`
   - `POST /predict` — accepts an uploaded image file, runs it through the model, and returns the predicted category and confidence score
2. Load the model once at startup with `mlflow.pyfunc.load_model("models:/food11@champion")`, not by loading a `.pth` file directly.
3. Read the mlflow tracking URI from an environment variable (e.g. `MLFLOW_TRACKING_URI`), defaulting to `http://127.0.0.1:5000`, so the value can be overridden from inside a container.

```bash
uv add fastapi uvicorn python-multipart
```

Run it locally first, before containerizing anything:

```bash
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

Test it with an image from your mini dataset:

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" http://127.0.0.1:8000/predict
```

> Question 3: Why load the model through an mlflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?

## Write the Dockerfile

Create a `Dockerfile` at the root of the repo using a multi-stage build:

- **Stage 1 (builder):** start from a Python base image, install `uv`, copy only `pyproject.toml` and `uv.lock`, then run `uv sync --frozen --no-dev` to build the virtual environment.
- **Stage 2 (runtime):** start from a slim Python base image, copy the built virtual environment from Stage 1, copy your `src/` folder, expose port `8000`, and set the container's entrypoint to run uvicorn.

> Question 4: Why copy `pyproject.toml`/`uv.lock` and run `uv sync` *before* copying the rest of the source code, instead of copying everything at once? What happens to the build cache when you only change a line in `serve.py`?

> Question 5: What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.

## Add a .dockerignore

Exclude everything the image build doesn't need: `.venv/`, `data/`, `mlruns/`, `mlflow.db`, `.git/`, `__pycache__/`, etc.

> Question 6: What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

## Build and run the image

```bash
docker build -t food11-api:latest .
```

Your mlflow tracking server is running on the host machine, not inside the container, so the container needs a way to reach it:

- **Linux:** `docker run --network host -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 food11-api:latest`
- **Mac/Windows:** `docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest`

> Question 7: Why can't the container simply use `127.0.0.1:5000` to reach the mlflow server on your host? What does `host.docker.internal` resolve to?

Test the containerized API the same way you tested it locally:

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" http://127.0.0.1:8000/predict
```

> Question 8: Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

## Commit your work

```bash
git add Dockerfile .dockerignore src/food11/serve.py pyproject.toml uv.lock
git commit -m "Containerize model serving with Docker"
git push
```

> Question 9: The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?


Question 1 :
The model was registered under the name food11 and was assigned version 1 (v1). The registered version was created from the logged model artifact of the best training run, victorious-wren-333.
A run’s logged model artifact is the trained model file produced by one specific experiment run. It remains associated with that run, including its parameters, metrics, and artifacts. A registered model is a separately managed, versioned entry in the MLflow Model Registry. It gives the model a shared name (food11), assigns versions (v1, v2, and so on), and makes it possible to use aliases such as champion to select which version should be served or deployed

Question 2
MLflow replaced the old built-in model stages, such as Staging and Production, with aliases. An alias is a custom, mutable name that points to a specific registered model version. In this project, the alias champion points to food11 version 1.
A model is versioned separately from its training run because a training run records the full experiment while the Model Registry manages models that are candidates for serving or deployment. This separation makes it possible to select, compare, approve, and deploy trained models independently of all experiment details.
An alias is more flexible than a fixed stage name because it can be reassigned to another version without changing application code. For example, if a better model is later registered as food11 version 2, the champion alias can be moved from version 1 to version 2. Any application loading:
models:/food11@champion
will then automatically use version 2 without changing the model name or deployment configuration. Your screenshot confirms that the champion alias is currently assigned to version 1

Question 3:
Loading the model through the MLflow URI models:/food11@champion is better than loading a .pth file directly because the API does not depend on a hard-coded local file path. MLflow resolves the URI through the Model Registry and loads the model version currently assigned to the champion alias. This makes the API independent of a specific training run, model file location, or fixed model version.

To serve a newer model version, I would register the new model as another version of food11, evaluate it, and move the champion alias to the new version. The API code would not need to change because it still loads models:/food11@champion. After restarting the API, it would load and serve the new model version.

Question 4:
Docker caches each layer based on the files that layer depends on, and reuses a cached layer instead of rerunning it as long as nothing it depends on has changed. By copying only pyproject.toml and uv.lock first and running uv sync (and the torch/torchvision install) before copying src/, that dependency-install layer only gets invalidated when the dependency files themselves change, not when I touch application code. In my build, resolving and installing all the base dependencies took over 5 minutes, and downloading the CPU torch/torchvision wheels took another 5 minutes on top of that. If I now change a single line in serve.py and rebuild, Docker reuses every cached layer up through both of those RUN steps and only reruns COPY src/ src/ and everything after it, which finishes in under a second. If I had copied the whole project first and run uv sync afterward, every source change would force a full dependency reinstall on every single rebuild.

Question 5:
I didn't build a separate naive single-stage image, but docker history on my actual image already shows where the size comes from. The final image is 2.02GB on disk (432MB of that counted as this image's own unique content, the rest shared with the base python:3.13-slim layers). Almost all of it is one layer: COPY --from=builder /app/.venv /app/.venv at 1.46GB, which is the CPU torch/torchvision wheels plus everything mlflow pulls in transitively (pandas, pyarrow, scikit-learn, scipy, matplotlib). Everything else is small in comparison: the debian base is 87.6MB, the Python install/apt cleanup layers add about 45MB combined, and my own src/ copy is 53KB. A naive single-stage build would keep that same 1.46GB venv layer, but it would also keep things the multi-stage build throws away: the uv binary I copied in to run the install, and the wheel/package cache uv builds up while resolving and downloading everything. None of that is huge on its own here, but it's dead weight that serves no purpose at runtime, and it's the difference between "everything needed to build the image" and "just what's needed to run it," which is the whole point of splitting into two stages.

Question 6:
Without a .dockerignore, docker build sends the entire build context to the Docker daemon before the build even starts, which slows down every single build and bloats the context transfer. In this project that would mean shipping over: .venv/ (my local virtual environment, separate from and irrelevant to the one built inside the image), data/ and backup_data/ (the Food-11 images, hundreds of MB), mlruns/ and mlflow.db (172MB+ of local run history), and .git/ (the whole repo history). None of these are referenced by a COPY instruction in my Dockerfile, so strictly speaking they wouldn't break the build, just make it noticeably slower and heavier to send on every rebuild. The one entry that could actually cause a problem if it weren't excluded is __pycache__/: if a stale .pyc compiled on my Windows machine got copied into the image via COPY src/ src/, it wouldn't necessarily crash anything, but it's exactly the kind of artifact that shouldn't be baked into a Linux container image built for a different Python build.

Question 7:
Every container gets its own network namespace with its own loopback interface, so 127.0.0.1 inside the container refers to the container itself, not my Windows host. My mlflow tracking server is a separate process running directly on my host machine, completely outside Docker, so if the containerized app tried to reach 127.0.0.1:5000 it would just be looking for a server running inside its own container and find nothing there. host.docker.internal is a special hostname that Docker Desktop resolves specifically to the host machine's internal address, from the container's point of view. That's why passing MLFLOW_TRACKING_URI=http://host.docker.internal:5000 into the container lets serve.py dial back out and reach the real mlflow server running on my machine.

Question 8:
Yes. I stopped the running container and started a brand new one from the exact same image, food11-api:latest, without rebuilding anything, and it loaded the champion model correctly and returned the exact same prediction (Fried food, confidence 0.7855) on the same test image as before. That confirms the model weights themselves are not baked into the image, only my code and Python environment are. At startup, serve.py calls mlflow.pyfunc.load_model("models:/food11@champion"), which reaches out over the network to the mlflow server (via host.docker.internal) and fetches whichever model version the champion alias points to at that exact moment. So the image is a fixed snapshot of the serving code and dependencies, but the actual model is fetched fresh every time a container starts. If I moved the champion alias to a different registered version and restarted a container from this same image, it would start serving the new model without any rebuild.

Question 9:
Right now the image only exists on my local Docker Desktop, it has never been pushed anywhere, so no other machine can pull it at all. To make it something a CI runner or a cluster could reliably use, I'd need to push it to an actual registry (Docker Hub, GHCR, or a private one) under a real name instead of just a local tag, and tag it with something more specific than :latest, like a git commit SHA, since :latest is mutable and can silently point to a different build later. On top of that, the image doesn't contain the model at all, it only knows how to fetch models:/food11@champion at startup, so whatever machine runs this container also needs network access to a real, reachable mlflow tracking server that has the food11 model registered with the champion alias already set, otherwise the container will start but fail to load a model.