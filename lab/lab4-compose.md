# Lab 4 - Orchestrating the app with Docker Compose

This lab continues the project from Lab 1 (git+dvc data), Lab 2 (training + mlflow tracking) and Lab 3 (a single Dockerfile serving the model). So far you've only ever run one container at a time, reaching your host's mlflow server through `host.docker.internal` or `--network host`. In this lab you will containerize mlflow itself and add a small web frontend, then bring all three services up together with Docker Compose: a **mlflow** service (tracking server + registry), an **inference** service (your Lab 3 API, now pointing at the mlflow container instead of your host), and a **frontend** service (a page where a human uploads an image and sees the prediction).

> What you need to know:
> - `docker compose` reads a `docker-compose.yml` and starts a set of *services* together, each built from its own image, on a private network it creates for you
> - inside that network, containers reach each other by **service name** (e.g. `http://mlflow:5000`) instead of `host.docker.internal` or a published port — Compose's embedded DNS resolves the name to the right container
> - a **named volume** is storage managed by Docker and kept outside any single container's filesystem; it's how you keep mlflow's database and artifacts alive across `docker compose down` / `up`, the same way `data/` is kept alive across `dvc checkout`
> - `depends_on` controls **start order**, not **readiness** — mlflow's container can be "started" before it's actually accepting connections, so the inference service still needs to handle a connection failure at startup
> - only the ports a human needs to reach directly should be `published` (mapped to the host); service-to-service traffic never needs a published port, since it stays inside the Compose network

## Before You Start: What You Need Installed

- Everything from Lab 3 already working: a Dockerfile that builds your inference API, and at least one model version registered in mlflow and moved to `Staging`.
- **Docker Compose** — bundled with Docker Desktop; on Linux, confirm the plugin is present with `docker compose version` (note: no hyphen, this is the newer `docker compose`, not the old standalone `docker-compose`).

## Project layout

You'll end up with a repo root that looks roughly like this:

```
./Dockerfile              # from Lab 3, the inference service image
./src/food11/serve.py     # from Lab 3
./mlflow/Dockerfile       # new: the mlflow tracking server image
./frontend/Dockerfile     # new: the frontend image
./frontend/app.py         # new: the Streamlit upload page
./docker-compose.yml      # new: wires the three together
```

## The mlflow service

### Dockerfile

Create `./mlflow/Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/mlruns"]
```

Note the path is under `/mlflow-data`, not `/mlruns` at the container root — this is the mount point for the volume you'll declare in `docker-compose.yml`, so the database and artifacts land on the volume instead of the container's writable layer.

> Question 1: What happens to everything written to `/mlflow-data` if you never mount a volume there and just `docker run` this image standalone? Try it: run the container, register nothing, stop it, remove it, start a new one from the same image — what do you see in the UI?

> Question 2: Why a named volume here instead of a bind mount to a folder in your repo (the way you might for local dev)? Would a bind mount work just as well?

## Update the inference service

Your Lab 3 `serve.py` already reads `MLFLOW_TRACKING_URI` from an environment variable, defaulting to `http://127.0.0.1:5000`. Nothing in the code needs to change — only the value you pass at container-run time will differ.

> Question 3: In Lab 3 you had to use `host.docker.internal` or `--network host` to reach mlflow from inside the container. In this lab, `MLFLOW_TRACKING_URI` will simply be `http://mlflow:5000`. Why does that hostname resolve now when it didn't before?

## The frontend service

### A minimal upload page

Create `./frontend/app.py`:

```python
import os
import requests
import streamlit as st

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")

st.title("Food-11 classifier")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, width=300)
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    response = requests.post(f"{INFERENCE_URL}/predict", files=files)
    if response.ok:
        result = response.json()
        st.write(f"**Prediction:** {result['category']} ({result['confidence']:.1%})")
    else:
        st.error(f"Inference service returned {response.status_code}: {response.text}")
```

> Question 4: Why does the frontend read `INFERENCE_URL` from an environment variable instead of hardcoding `http://inference:8000`? Think about what happens if you ever `docker run` this frontend image on its own, outside Compose.

### Dockerfile

Create `./frontend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir streamlit requests

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## Wire it together with Docker Compose

Create `./docker-compose.yml` at the repo root:

```yaml
services:
  mlflow:
    build: ./mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow-data

  inference:
    build: .
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      INFERENCE_URL: http://inference:8000
    depends_on:
      - inference

volumes:
  mlflow-data:
```

> Question 5: Only `mlflow` and `frontend` publish a port to the host. `inference` doesn't. Why not, and how does the frontend still reach it?

> Question 6: `depends_on` here only waits for the mlflow *container process* to start, not for the tracking server inside it to be ready to accept connections. If your `serve.py` tries to load the Staging model at startup and mlflow isn't ready yet, what happens to the `inference` container? Look at `docker compose logs inference` if it fails.

## Bring the stack up

```bash
docker compose up --build
```

Open the mlflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) and the frontend at [http://127.0.0.1:8501](http://127.0.0.1:8501). Upload an image and confirm you get a prediction back.

> Question 7: Run `docker compose ps`. Which services have a published port listed, and which don't? Does that match what you'd expect from the `docker-compose.yml`?

## Promote a new model version and reload

In the mlflow UI, register a new version of your model (or re-register the same run under a new version) and move it to `Staging`, replacing the version currently there.

> Question 8: Refresh the frontend and upload an image again. Does the prediction come from the new model version, or the old one? Your `serve.py` loads the model once, at startup — what single command lets you pick up the new Staging version without rebuilding any image?

Try it:

```bash
docker compose restart inference
```

> Question 9: Why does `restart` alone work here — no rebuild needed? What does that tell you about what's baked into the inference image versus fetched at container startup?

## Persistence check

```bash
docker compose down
docker compose up
```

> Question 10: Is your registered model and its Staging assignment still there after this `down`/`up` cycle? Now try `docker compose down -v` followed by `docker compose up` — what's different this time, and why?

## Commit your work

```bash
git add mlflow/Dockerfile frontend/Dockerfile frontend/app.py docker-compose.yml
git commit -m "Orchestrate mlflow, inference and frontend with Docker Compose"
git push
```

> Question 11: This compose file is still meant to run on one machine. What would have to change for the `inference` service to run as three replicas behind a load balancer, or for the mlflow service to survive a machine failure? (You don't need to implement this — just name what Docker Compose can't give you here.)

I first built the MLflow image and ran it without mounting a volume. I created a test experiment called no-volume-test and confirmed that it existed. Then I removed that container and started a new one from the same image. The new container showed only the Default experiment—the test experiment was gone.

That’s because the database file, sqlite:////mlflow-data/mlflow.db, was stored in the container’s writable layer. Without a volume mounted at /mlflow-data, removing the container also removed that layer and the data in it. The image itself wasn’t changed; the experiment existed only in the deleted container.

Question 2
A named volume is managed by Docker rather than tied to a particular folder on my computer. That makes the same docker-compose.yml easier to use on Windows, macOS, or a CI runner: Docker creates and mounts the volume wherever it stores its data.

A bind mount to a folder in my repository would also work for local development. It would make files like mlflow.db and mlruns/ visible on my machine, which can be useful for inspecting or backing them up. But it also ties the setup to a host path and can behave differently across operating systems, particularly around permissions and file locking. Since portability matters here, a named volume is the better default.

While setting up the stack, I also ran into two issues with my MLflow 3.16.x installation.

First, MLflow rejected requests with a 403 Invalid Host header error. This happened whether the request came from the inference service using mlflow, or from my host machine using 127.0.0.1:5000. The newer MLflow version validates Host headers, so I added the relevant names to the MLflow server’s --allowed-hosts option:

mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000
The second issue appeared when the inference service tried to load a registered model and failed with No such artifact: ''. The lab’s --default-artifact-root /mlflow-data/mlruns setting made MLflow record a direct filesystem path as the artifact location. Clients then tried to access that path on their own machines instead of requesting the artifacts through the MLflow server. In my case, artifacts ended up under C:\mlflow-data on Windows rather than in the Docker volume.

I fixed that by setting the client-facing artifact root to mlflow-artifacts:/ and using --artifacts-destination /mlflow-data/mlruns for the server’s actual storage. This lets clients access artifacts through MLflow’s HTTP API while the server stores the files in the mounted volume.

Question 3
http://mlflow:5000 works because Docker Compose connects the services to a shared private network and provides DNS for service names. The inference container can resolve mlflow and reach the MLflow container directly.

host.docker.internal solves a different problem: it lets a container reach a service running on the host computer. I needed it in Lab 3, when MLflow was running directly on Windows rather than in Docker. Once MLflow became a Compose service, the containers could communicate over their shared network, so I no longer needed host.docker.internal.

Question 4
If the frontend URL were hardcoded as http://inference:8000, the image would depend on the Compose network, where inference is a known service name. Running the frontend by itself with plain docker run would fail because Docker wouldn’t know what host inference refers to outside that network.

Using an environment variable instead makes the frontend image reusable. At runtime, I can point it to the Compose service, a local inference server, or a remote endpoint without changing the code or rebuilding the image.

Question 5
Only mlflow and frontend publish ports because those are the services people need to reach directly: MLflow’s UI and the frontend’s upload page.

The frontend calls inference over the private Compose network using http://inference:8000. That communication doesn’t require publishing the inference port to the host. Leaving it unpublished avoids opening a port that outside clients don’t need.

Question 6
I saw this failure after running docker compose down -v, which deleted the MLflow data volume. When I started the stack again, the inference container started as soon as the MLflow process started, but there was no longer a registered food11 model to load.

The inference app immediately failed with Registered Model with name=food11 not found while trying to load the champion model. depends_on controls startup order, but it doesn’t wait for a model to be registered. Since serve.py doesn’t retry or wait when loading the model fails, FastAPI exited. There was no restart policy configured, so the container stayed stopped. The traceback in docker compose logs inference showed the missing model as the cause.

Question 7
In docker compose ps, mlflow showed 0.0.0.0:5000->5000/tcp, and frontend showed 0.0.0.0:8501->8501/tcp. Those mappings mean their ports are published on the host.

The inference service showed only 8000/tcp, with no host-port mapping. That matches the Compose configuration: the service makes port 8000 available to other containers on the Compose network, but doesn’t publish it to my computer.

Question 8
After I registered a new model version and moved the champion alias, refreshing the frontend didn’t change the predictions. That’s because serve.py loads the model once when the inference process starts and keeps it in memory.

To make inference load the model version that the alias currently points to, I ran:

bash
docker compose restart inference
The restarted process loads the model again from models:/food11@champion. I didn’t need to rebuild the image.

Question 9
Restarting inference works without a rebuild because the image contains the serving code and its Python dependencies, but not the model weights. At startup, the container fetches the model from MLflow based on the version currently assigned to the champion alias.

I confirmed this by deleting the Compose volume, training a new model, and registering it. Restarting inference with the same image loaded the new model without a rebuild. The image is a fixed snapshot of the code and environment; the model is fetched from MLflow each time the container starts.

Question 10
After running docker compose down—without -v—and bringing the stack back up, the food11 model and its champion alias were still there. The named volume remained intact, so MLflow kept its database and artifacts even though the containers and network had been removed.

After running docker compose down -v, the result was different. The named volume was deleted, so MLflow’s database and artifacts were gone. The registry was empty, and inference failed because the food11 model no longer existed. I had to train and register a model again, then assign the champion alias, before the stack could serve predictions.

Question 11
Docker Compose runs containers on a single Docker engine, so it can’t spread three inference replicas across different machines or provide cluster-wide load balancing. For that, I’d need a multi-node orchestrator such as Docker Swarm or Kubernetes, along with a load balancer or ingress controller to distribute requests.

MLflow also needs its state moved off the single machine. In this setup, the SQLite database and named artifact volume both live on local disk, so a machine failure could take both out. A more resilient setup would use an external database, such as PostgreSQL or MySQL, with its own replication or failover, and object storage such as S3 or GCS for artifacts. Compose is useful for connecting services on one machine, but it doesn’t provide high availability across a cluster.