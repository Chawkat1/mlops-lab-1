# MLOps Lab 1 — Food11 Data Pipeline

prepared by chawkat choumane grp 6 (231736)

MLOps lab project demonstrating data versioning with Git, DVC, and DagsHub, using the Food11 dataset.

Project Structure
.
├── data/
│   └── food11_raw/          # Raw images, organised by split
│       ├── training/
│       ├── evaluation/
│       └── validation/
├── src/
│   └── food11/
│       └── data.py          # Preprocessing script
├── data.dvc                 # DVC pointer file for the tracked data
├── pyproject.toml           # Project dependencies (managed by uv)
├── uv.lock
└── README.md

Raw images are named <label></label>_<id></id>.jpg, where <label></label> is a numeric category ID (0–10) mapped to a Food11 category name (e.g. 0 → Bread, 2 → Dessert).

Setup

Install uv if you don't have it:

bash
pip install uv

Clone the repo and install dependencies:

bash
git clone https://github.com/Chawkat1/mlops-lab-1.git
cd mlops-lab-1
uv sync

Pull the tracked data from the DVC remote (DagsHub):

bash
dvc pull
Data Preprocessing

src/food11/data.py reads the raw dataset and produces two processed versions under data/:

food11_processed — full dataset, images resized to 128×128 and sorted into per-category subfolders (e.g. training/Bread/, training/Dessert/, ...).
food11_processed_mini — same structure, capped at 100 images per category per split, for fast local development and testing.

Run it with:

bash
uv run python ./src/food11/data.py

These processed folders are intended for local use only and are not tracked by DVC or pushed to the remote (see .gitignore).

Data Versioning

Raw data is tracked with DVC and stored on DagsHub:

bash
dvc add data
git add data.dvc
git commit -m "Update tracked data"
git push
dvc push

To inspect or restore an older version of the data:

bash
git log --oneline -- data.dvc   # list commits that changed the data pointer
git checkout <commit-hash></commit>
dvc checkout

Return to the latest version:

bash
git checkout main
dvc checkout
Known Limitations
dvc push to the DagsHub remote was unreliable during development, frequently failing with network timeouts/connection drops on larger pushes. As a workaround, the tracked raw dataset was reduced to a small sample (5 images per category per split) to keep the pipeline demonstrable end-to-end. The full-size dataset can be regenerated locally by pointing data.py at a complete copy of the raw data.

the answers of the questions are in the lab/lab1.md
