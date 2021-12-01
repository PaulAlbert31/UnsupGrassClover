# UnsupGrassClover
Official implementation of "Utilizing unsupervised learning to improve sward content prediction and herbage mass estimation"

# Running the code
train.sh contrains the commands to train the unsupervised algorithm and to use the learned weights to initialize the regression network

# Datasets
The irish dataset is not publicaly available
The GrassClover dataset and unlabeled images can be found at https://competitions.codalab.org/competitions/21122#participate-get-data

You might want to preprocess the dataset by downscaling the unsupervised images to a lower resolution before training to speed up the image operations.