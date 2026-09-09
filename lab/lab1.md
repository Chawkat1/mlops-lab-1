## Question 1

After running `uv init`, the project contains files such as `pyproject.toml`, `.python-version`, and `README.md`. The `pyproject.toml` file stores the project information and Python dependencies. `.python-version` specifies the Python version to use, while `README.md` is used to document the project.

## Question 2

After running `dvc init`, DVC creates the `.dvc` folder and the `.dvcignore` file. The `.dvc` folder contains DVC configuration and local cache information, while `.dvcignore` tells DVC which files to ignore. We should push `.dvc` and `.dvcignore` to Git, but not the DVC cache or large data files.

## Question 3

Using `--global` stores the Dagshub credentials in the global DVC configuration of my computer, outside the project repository. Other options are `--project`, `--local`, and `--system`. Credentials must never be pushed to GitHub because they are private secrets; only the non-secret remote configuration in `.dvc/config` should be committed.

## Question 4

After running `dvc add data`, DVC updated the `.gitignore` file by adding the `data/` folder. This prevents Git from tracking and uploading the actual dataset, which may be large. The data is now managed by DVC instead of Git


## Question 5

Yes, a file named `data.dvc` should appear in the project root. It is a small metadata file created by DVC. It contains information about the tracked `data/` folder, including its path, size, and a hash that identifies the exact version of the data. The `data.dvc` file should be committed to Git, while the actual `data/` folder should not.


## Question 6

On the GitHub main branch, the source code and project files are present, but the actual `data/` folder is not uploaded because it is ignored by Git. Instead, GitHub contains the `data.dvc` file, which is a small pointer file that identifies the tracked version of the dataset.

On Dagshub, the actual dataset is present because the command `dvc push` uploaded the DVC-tracked data from the local cache to the configured Dagshub remote



## Question 7

After cloning the GitHub repository into a new folder, the `data/` folder is not present. This is because GitHub contains only the `data.dvc` pointer file, not the actual dataset.

To download and restore the dataset from the Dagshub DVC remote, the required command dvc pull.

This command reads the information in `data.dvc`, downloads the corresponding data from Dagshub, and recreates the `data/` folder locally.


## Question 8

After checking out commit `85dbd98` and running `dvc checkout`, only the `food11_raw` folder was present in the `data` directory. The processed datasets were not present because this older commit was created before `food11_processed` and `food11_processed_mini` were generated. This shows that Git restores the code and DVC pointer file, while `dvc checkout` restores the corresponding version of the data.
