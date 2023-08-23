# Sleep Apnea Segmentation

An exercise to predict sleep apnea events based on the polysomnographic records of patients with sleep apnea. 
The following database is used: (https://physionet.org/content/ucddb/1.0.0/).


The model used for training is [Unet1D](https://github.com/apneal/segmentation_models/tree/master) taken from the `segmentation_models` repository.

The project is organized into different directories:

- `notebooks`: Contains Jupyter notebooks used for exploring the data, training, and evaluation.
- `src`: Contains utility scripts required for data processing and visualization.
- `models`: This is where trained models are saved during training.
- `data`: This folder holds the necessary data for the project.
