
# Lab 2 - Model training and experiment tracking with MLflow

This lab continues the project started in Lab 1. You now have a git+dvc repo with the raw and processed Food-11 datasets tracked. In this lab you will write the training code, run a local MLflow tracking server, log parameters and metrics for each training run, and compare several runs in the MLflow UI.

Question 1:
After installing the training libraries, pyproject.toml was updated to include mlflow, torch, torchvision, and scikit-learn in the project dependencies.
The uv.lock file was updated with the exact versions of these libraries and all their required dependencies. This allows the same Python environment to be reproduced on another machine.

Question 2
--backend-store-uri sqlite:///mlflow.db tells MLflow where to store the experiment metadata. In this lab, it uses a local SQLite database file called mlflow.db. It contains experiment names, run IDs, parameters, metrics, timestamps, and run status.

--default-artifact-root ./mlruns tells MLflow where to save the artifacts produced by each run. In this lab, they are saved inside the local mlruns/ folder. Artifacts include trained models, model weights, images, plots, and other output files.

Question 3
mlflow.db and mlruns/ should not be tracked by Git because they are generated local outputs that change after every training run. Tracking them would make the repository large, cluttered, and constantly changing.

They should not be tracked by DVC either because they are MLflow’s local experiment-tracking database and artifact storage, not the versioned Food-11 dataset. DVC is used here to version the dataset, while MLflow is used to record runs, parameters, metrics, and trained-model artifacts.

Question 4
When mlflow.set_experiment("food11") is called for the first time and no experiment with that name exists, MLflow automatically creates a new experiment called food11. It does not raise an error. The terminal confirmed this with the message:

text
Experiment with name 'food11' does not exist. Creating a new experiment.
Afterward, the new food11 experiment appeared in the MLflow UI and was ready to contain training runs.

Question 5
mlflow.log_param() stores a fixed configuration value for a training run, such as the learning rate, batch size, number of epochs, model architecture, or dataset name. These values are normally chosen before training starts and do not change during the run.

mlflow.log_metric() stores measured results produced during or after training, such as training loss, validation loss, validation accuracy, and test accuracy. These values can change at every epoch.

log_metric() takes a step argument because MLflow needs to know at which point in training the value was measured—for example, at epoch 1, 2, 3, and so on. This allows MLflow to draw charts showing how loss and accuracy evolve over time. log_param() does not need a step because each parameter is a single fixed value for the entire run.

Question 6
In the MLflow UI, the successful run agreeable-cod-892 contains the logged parameters dataset, epochs, lr, batch_size, model, num_classes, and image_size. The Model metrics tab shows charts for train_loss, val_loss, val_accuracy, and test_accuracy. In this run, training loss decreased throughout training, while validation accuracy reached approximately 0.57 and final test accuracy was approximately 0.57. The run also contains a logged model named model, with status Ready. The model artifact is stored locally under the configured artifact root, in a path similar to mlruns/<experiment_id>/<run_id>/artifacts/model.

Question 7
After comparing the four runs in MLflow, the learning rate 0.0001 gave the best validation accuracy, reaching 0.707. This run also obtained the best test accuracy, 0.739. A higher learning rate is not always better: lr=0.01 performed poorly, with validation accuracy 0.161 and test accuracy 0.156, likely because the learning updates were too large. For this pretrained ResNet-18 fine-tuning experiment, the smaller learning rate was more stable and produced the best result.

Question 8
The parallel coordinates plot shows that the best validation accuracy is associated with lr=0.0001 and batch_size=32. The largest learning rate, 0.01, produced the weakest result and highest training loss. With lr=0.001, changing the batch size from 32 to 64 slightly reduced validation accuracy from 0.513 to 0.494. Therefore, the learning rate had a stronger impact on performance than the batch size in these four experiments.

Question 9
After sorting the runs by val_accuracy in descending order, the best run was victorious-wren-333, with lr=0.0001, batch_size=32, and val_accuracy=0.707. Its run ID is: b0895843421943b88497c9585158c840