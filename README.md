# noisy-drone-detection-benchmark
Scripts to load and inspect the drone detection benchmark dataset

## Dataset
The dataset is available at [kaggle](https://www.kaggle.com/sgluege/noisy-drone-data-benchmark). Download the dataset and place it in a subfolder `dataset/`. 

It comes in the form of 3 files:
- `class_stats.csv`: contains the number of samples per class
- `SNR_stats.csv`: contains the number of samples per SNR
- `dataset.pt`: contains the dataset itself

## Load and inspect the dataset
Use the script `load_dataset.py` to load the dataset using a custom torch Dataloader. It also plots a sample of the dataset which should look like this: 
![sample_input_data.jpg](doc/img/sample_input_data.jpg)
