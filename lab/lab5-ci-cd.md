# Lab 5 - CI/CD with GitHub Actions

This lab continues the project from Labs 1–4. So far every step (testing, building the image, choosing the model to serve) was done by hand from your laptop. In this lab you will make GitHub do it for you: on every push, a workflow lints your code, runs unit and data tests, builds the Docker image, checks that the container starts, and (on `main` only) pushes the image to the GitHub Container Registry tagged with the commit SHA. You will also write a small promotion script that applies a quality gate before moving the `champion` alias.

> What you need to know:
> - a **workflow** is a YAML file under `.github/workflows/`; an **event** (push, pull request, schedule, manual click) starts it; it contains **jobs** that run on fresh virtual machines called **runners**; each job is a list of **steps**
> - a step is either a shell command (`run:`) or a reusable **action** (`uses:`), such as `actions/checkout` or `docker/build-push-action`
> - every run starts from an empty machine: nothing from a previous run is there unless you cache it
> - **secrets** are stored in the repo settings and injected at run time; `GITHUB_TOKEN` is a short-lived token GitHub creates for each run, and its rights are set with `permissions:`
> - CI checks that the code, the data contract and the container are healthy. It does **not** train the real model: there is no GPU and the run would take hours
> - the image and the model are released independently: the image is tagged with the **git commit SHA**, the model is selected with the registry **alias** `champion`

## Before You Start

- Labs 1–4 working: `src/food11/data.py`, `train.py`, `serve.py`, the `Dockerfile` and `docker-compose.yml` pushed to your GitHub repo.
- Your serving code reads `MLFLOW_TRACKING_URI` from the environment (Lab 3).

## Make the code testable

Unit tests must run in a second, without a GPU, without data from DVC and without a running MLflow server. That requires two small refactorings.

### Pure functions in data.py

Make sure `src/food11/data.py` exposes at least these two functions (and that the copy loop only runs under `if __name__ == "__main__":`):

```python
CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
           "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'"""
    ...

def resize_image(img: "PIL.Image.Image", size: int = 128) -> "PIL.Image.Image":
    """Returns an RGB image of size x size."""
    ...
```

### A model that can be replaced in tests

In `src/food11/serve.py`, keep the loaded model in a module-level variable named `model`, and only load it from MLflow if the environment variable `SKIP_MODEL` is not set:

```python
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.environ.get("SKIP_MODEL"):
        return
    model = mlflow.pyfunc.load_model("models:/food11@champion")
```

`/health` should still return 200 when `SKIP_MODEL` is set; `/predict` can return 503 if `model is None`.

> Question 1: Why is it a bad idea for a unit test to load the real model from your MLflow server? List at least two ways the test could fail even though your code is correct.

## Write the tests

```bash
uv add --dev pytest ruff httpx
mkdir -p tests/fixtures
```

Copy one small image from `food11_processed_mini` into `tests/fixtures/bread.jpg` and commit it (it is a few kB, so it goes in git, not DVC).

### Unit tests for the data code

Create `tests/test_data.py`:

```python
from PIL import Image
from src.food11.data import category_from_filename, resize_image, CLASSES

def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"

def test_resize_image_shape_and_mode():
    img = Image.new("L", (512, 384))          # a grayscale, non-square image
    out = resize_image(img)
    assert out.size == (128, 128)
    assert out.mode == "RGB"

def test_eleven_classes():
    assert len(CLASSES) == 11
```

### A test of the API with a fake model

Create `tests/test_api.py`. The fake model returns fixed probabilities so the test checks *your* code (decoding the image, building the JSON), not the model:

```python
from fastapi.testclient import TestClient
import src.food11.serve as serve

class FakeModel:
    def predict(self, x):
        return [[0.90] + [0.01] * 10]

def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200

def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as f:
            r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0
```

Adapt `FakeModel.predict` to whatever your `serve.py` expects back from the model (a list, a numpy array, a DataFrame...).

```bash
uv run ruff check src tests
uv run pytest -q
```

> Question 2: The fake model always answers "Bread" with 0.90 confidence. What exactly does `test_predict_returns_valid_json` prove, and what does it *not* prove?

## Validate the data against a contract

Create `src/food11/validate_data.py`. Given a processed dataset folder (`data/food11_processed` or `data/food11_processed_mini`), it must check and exit with a non-zero status code if anything fails:

1. each split (`training`, `validation`, `evaluation`) contains exactly the 11 class folders
2. every image opens with PIL, is RGB and is 128×128
3. every class has at least `--min-per-class` images (default 50 for the mini dataset)
4. no image appears in two splits: compare a hash of the file content (`hashlib.md5(path.read_bytes()).hexdigest()`)
5. for each class, its share of the training split is within `--tau` (default 0.05) of its share in the validation split

Print a short report (one line per check, PASS/FAIL) and finish with `sys.exit(1)` if any check failed.

```bash
uv run python src/food11/validate_data.py data/food11_processed_mini
```

Now break the contract on purpose: copy one image from `training/Bread` into `evaluation/Bread`, and run the script again. Then remove the copy.

> Question 3: Which check caught it? Why is a train/test duplicate more dangerous than a corrupted image, even though the corrupted image makes the training script crash and the duplicate does not?

> Question 4: Your CI runner cannot see your data (it is in DVC, and maybe in a local remote). Where in your workflow should `validate_data.py` run so that it still protects you? Give two places.

## The CI workflow

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check src tests
      - run: uv run pytest -q

  smoke:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t food11-api:ci .
      - name: start the container without a model and probe it
        run: |
          docker run -d --name api -p 8000:8000 -e SKIP_MODEL=1 food11-api:ci
          for i in $(seq 1 20); do
            curl -sf localhost:8000/health && exit 0
            sleep 2
          done
          docker logs api
          exit 1
```

Commit and push. Open the **Actions** tab of your repo and watch the run.

> Question 5: The `smoke` job has `needs: test`. What happens to it if a unit test fails? What would happen without `needs`?

> Question 6: How long did the whole run take? Run it a second time (push an empty commit with `git commit --allow-empty -m "ci"`) and compare. Which step got faster, and why?

> If your `uv sync` downloads the multi-GB CUDA build of torch, the run will be slow. Use the CPU-only index from Lab 2 in `pyproject.toml`.

## Push the image to the registry

So far the image only exists inside the runner, and is thrown away at the end. Add a third job that runs **only on `main`**, logs in to the GitHub Container Registry (ghcr.io) with the run's `GITHUB_TOKEN`, and pushes the image tagged with the commit SHA:

```yaml
  push:
    needs: smoke
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

> The image name must be lowercase. If your GitHub username has capital letters, replace `${{ github.repository }}` with your lowercase `<username>/<repo>`.

Push to `main`, wait for the run, then find the image under **Packages** on your GitHub profile. Pull it on your laptop:

```bash
docker pull ghcr.io/<username>/<repo>:<commit-sha>
```

> Question 7: This answers Question 9 of Lab 3. Why tag with the commit SHA rather than `latest`? Describe a concrete situation where `latest` leads two machines to run different code.

> Question 8: Why does the `push` job declare `permissions:` explicitly? What could a compromised step do with a token that had write access to everything in the repo?

## Protect main with the pipeline

In the repo settings, add a **branch protection rule** (or ruleset) for `main`: require a pull request before merging, and require the `test` and `smoke` checks to pass.

Then, on a new branch, break a test on purpose (for example, make `category_from_filename` return the wrong class), push the branch and open a pull request.

> Question 9: What does the pull request page show? Could you still merge? What does that change for a team compared to "everybody runs the tests on their laptop before pushing"?

Fix the test, push again, and merge once the checks are green.

## A quality gate for the model

The image is now built and shipped by the pipeline. The model is still chosen by hand in the MLflow UI. Write `src/food11/promote.py` that applies the gate from the lecture:

1. take a candidate model version number as argument (`--version 8`)
2. read the `test_accuracy` metric (or the name you used in Lab 2) of the run that produced the candidate, and of the run that produced the current `champion` (use `MlflowClient().get_model_version(...)` / `get_model_version_by_alias(...)`, then `get_run(run_id).data.metrics`)
3. compute the standard error of the champion's accuracy, `se = sqrt(p * (1 - p) / n)`, with `n` the size of your test split
4. if `acc_candidate >= acc_champion + 2 * se`: move the `champion` alias to the candidate; otherwise set the `challenger` alias on it and leave `champion` where it is
5. in both cases, write the decision and both accuracies as **tags** on the candidate version (`client.set_model_version_tag(...)`)

Train two new runs (for example one with a very small learning rate, one with your best settings), register both, and run the script on each.

```bash
uv run python src/food11/promote.py --version <n>
```

> Question 10: For each candidate, what did the gate decide, and with which numbers? With your test-set size, how many accuracy points does `2 * se` represent?

> Question 11: `promote.py` needs to reach your MLflow server, so it cannot run in GitHub's cloud runners while MLflow lives on your laptop. Name two ways to make it run automatically after training. (Hint: DagsHub also offers a hosted MLflow server per repository.)

## Commit your work

```bash
git add .github/workflows/ci.yml tests/ src/food11/validate_data.py src/food11/promote.py src/food11/data.py src/food11/serve.py pyproject.toml uv.lock
git commit -m "Add CI/CD pipeline, tests, data contract and promotion gate"
git push
```

> Question 12: Draw (in `lab/lab5.md`) the pipeline you now have, from `git push` to a running container. Mark which steps are automatic, which still need a human, and where the code version and the model version are decided.


---

## My answers

Question 1
A unit test should only check my code, and it should give the same result every time. If it loads the real model from my MLflow server, it can fail for reasons that have nothing to do with my code:
- The MLflow server is not running, or the address is wrong. On a CI runner there is no server at all.
- The `champion` alias was moved to another version, so the model is different from yesterday.
- The model download is big and slow, so the test becomes slow or hits a timeout. The internet can also drop in the middle.
- Someone deleted the model or the artifacts, or the server needs a login that the test does not have.

Because of this I added `SKIP_MODEL`, and the tests use a small fake model instead.

Question 2
`test_predict_returns_valid_json` proves that my API code works: it reads the uploaded file, turns it into the right tensor, calls `model.predict`, applies softmax, picks the class name from the index and returns JSON with `category` and `confidence`, with the status 200 and a confidence between 0 and 1.

It does not prove that the real model is good, or even that it is the right model. The fake model always says Bread, so the test cannot tell me if the real model predicts the right food. It also does not check that the real model accepts this input shape, that the class order in `CLASS_NAMES` matches the training order, or that loading from MLflow works.

Question 3
The check "no image appears in two splits" caught it. It compares the md5 hash of the file content, so the copy of the Bread image in `evaluation/Bread` had the same hash as the one in `training/Bread`. The check printed FAIL and the script exited with code 1. After I removed the copy, the number of duplicates went back to what it was before.

One more thing I found: this check already fails on my mini dataset without my test copy. There are 38 images that are identical in two different splits. This comes from the original dataset, not from my code, so I did not change the data. I only noted it.

A duplicate between train and test is more dangerous than a corrupted image because it is silent. A corrupted image makes the training crash, so I see the problem right away and fix it. A duplicate does not crash anything. The model has already seen the test image during training, so it gets it right just by remembering it. The test accuracy looks better than it really is, and I could ship a weaker model thinking it is a good one without noticing.

Question 4
The runner cannot see my data, so `validate_data.py` has to run where the data is available. Two places:
1. At the end of the data preparation or training step, on my own machine (or on the machine that does `dvc pull`). The script runs right after `data.py` creates the processed folders, and if it fails the training does not start.
2. In a CI job that first does `dvc pull` from a remote the runner can reach (for example DagsHub or an S3 bucket, with the credentials stored as GitHub secrets), and then runs the script. It could also run as a scheduled workflow, so a bad change in the data is found even when no code changed.

Question 5
`needs: test` means `smoke` only starts after `test` finished successfully. If a unit test fails, `smoke` is skipped, so GitHub does not waste time building a Docker image from code that is already known to be broken.

Without `needs`, the jobs run at the same time in parallel. The Docker build would start even if the tests fail, and the run could show a green container check next to a red test job.

Question 6
The first run took about 4 minutes. After that I pushed an empty commit (`git commit --allow-empty -m "ci"`) and the second run took 1 minute 54 seconds, so about half the time.

Two steps got faster:
- `uv sync` went from about 25 seconds to about 15 seconds. `enable-cache: true` keeps the downloaded packages (torch is big) and the next run takes them from the cache instead of downloading them again.
- The `push` job went from about 2 minutes 20 seconds to about 23 seconds. The `docker/build-push-action` step itself only took 4 seconds in the second run, because `cache-from: type=gha` lets Docker reuse the image layers saved by the first run, and nothing had changed in the code.

The `smoke` job did not get faster because it runs a plain `docker build` with no cache. Every run still starts on an empty machine, so the cache is the only thing that carries over.

Question 7
The commit SHA points to exactly one version of the code, and it never changes. If something breaks I know which commit made the image, and I can go back to the exact previous image. `latest` is just a label that moves every time someone pushes.

Example with `latest`: I push a new image on Monday. A server pulled `latest` on Friday last week and has been running since then, and a second server starts today and pulls `latest`. Now the two machines run different code under the same name, and nobody can tell it by looking at the tag. With the SHA tag, each machine says exactly which version it is running.

Question 8
By default a token can have more rights than the job needs. With `permissions:` I give the `push` job only `contents: read` and `packages: write`, so it can read the code and push the image and nothing else (least privilege).

If a step is compromised (for example a bad third-party action, or a malicious dependency), it can use the token. With write access to everything in the repo it could push code to `main`, change the workflow files, edit or delete releases, and steal or change other secrets. With the small permissions, the damage stays limited to the package.

Question 9
I changed `"0": "Bread"` to `"0": "Egg"` in `data.py` on a branch called `break-test`, pushed it and opened pull request #1. The PR page showed "Some checks were not successful: 2 failing, 4 skipped checks".
- `ci / test` failed with a red cross (it showed up twice, once for the `pull_request` event and once for the `push` event, because my workflow runs on both). Both were marked **Required**.
- `ci / smoke` and `ci / push` were skipped, because they need `test` to pass first. So the broken code never got built into an image.
- The **Merge pull request** button was greyed out, so I could not merge. The branch rule blocks it until the required checks are green.

For a team this changes a lot. With "everybody runs the tests on their laptop", nothing forces anyone to do it, people forget, and every laptop is a bit different (python version, packages, data). With the pipeline the same checks run on a clean machine for every change, and nobody can merge a broken change by accident.

Question 10
- Champion: version 2, test accuracy 0.7728. The test split has n = 1096 images, so se = sqrt(0.7728 * (1 - 0.7728) / 1096) = 0.0127. The bar to beat is 0.7728 + 2 * 0.0127 = 0.7981.
- Version 3 (learning rate 1e-7, the very small one): test accuracy 0.1122. This is far below 0.7981, so the gate decided **rejected**. `challenger` was set on it and `champion` stayed on v2. The model almost did not learn (about 11 percent, close to random for 11 classes).
- Version 4 (learning rate 1e-4, my best settings): test accuracy 0.7591. This is also below 0.7981, so the gate decided **rejected** too, and `challenger` moved to version 4. It is even a bit lower than the champion, so it does not beat it.

`2 * se` = 2 * 0.0127 = 0.0254, so about 2.5 accuracy points. A new model needs to be at least 2.5 points better than the champion, otherwise the difference could just be noise from the small test set. Both decisions and the accuracies are saved as tags on the model versions (`gate_decision`, `candidate_accuracy`, `champion_accuracy`, `gate_threshold`).

Question 11
Two ways:
1. Put MLflow on a server that GitHub can reach, for example the hosted MLflow of DagsHub (one per repository). Then a GitHub Actions workflow can run `promote.py` with the tracking URI and the token stored as secrets, for example right after the training step, or when I click "run workflow" manually.
2. Keep MLflow on my machine but run `promote.py` on my machine at the end of the training itself, either by calling it as the last line of a training script or with a self-hosted runner installed on my computer so the workflow runs there and can reach `127.0.0.1:5000`.

Question 12
My pipeline from `git push` to a running container:

```text
 [HUMAN] write code, run tests quickly, git push
        |
        v
 +------------------------ GitHub Actions (AUTOMATIC) ------------------------+
 |  test:  uv sync -> ruff -> pytest        (every push and every PR)          |
 |    |                                                                        |
 |    v   needs: test                                                          |
 |  smoke: docker build -> run with SKIP_MODEL=1 -> curl /health               |
 |    |                                                                        |
 |    v   needs: smoke, only on main                                           |
 |  push:  login ghcr.io -> build -> push image tagged with the commit SHA     |
 +-----------------------------------------------------------------------------+
        |
        v
 ghcr.io/chawkat1/mlops-lab-1:<commit-sha>        <-- CODE VERSION is decided here
        |
        v
 [HUMAN] docker pull / docker compose up with that SHA     (still manual)
        |
        v
 running container (inference) --- at startup loads models:/food11@champion
                                                      ^
                                                      |
 [HUMAN] train.py -> register model -> promote.py --version N
         (gate: accept if candidate >= champion + 2 * se)
         MODEL VERSION is decided here (the alias `champion`)
```

- Automatic: lint, unit tests, data/API tests, Docker build, smoke test, pushing the image to the registry.
- Still done by a human: writing the code, merging the pull request, training the model, running `promote.py`, and pulling or restarting the container on the machine that serves it. Branch protection is set up once by a human, then it is automatic.
- Code version: decided by the git commit, which becomes the image tag (the SHA).
- Model version: decided by the MLflow alias `champion`, moved only if the quality gate says yes. After the alias moves, the container must be restarted to load the new model (like in Lab 4).
