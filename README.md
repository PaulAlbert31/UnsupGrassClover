# UnsupGrassClover
Official implementation of "Utilizing unsupervised learning to improve sward content prediction and herbage mass estimation" [paper](https://arxiv.org/pdf/2204.09343.pdf). Published in the European Grassland Federation General Meeting.

# Running the code
train.sh contrains the commands to train the unsupervised algorithm and to use the learned weights to initialize the regression network

# Datasets
The irish dataset is not publicaly available
The GrassClover dataset and unlabeled images can be found at https://competitions.codalab.org/competitions/21122#participate-get-data

You might want to preprocess the dataset by downscaling the unsupervised images to a lower resolution before training to speed up the image operations.

# Cite our research
If it helped your work

```
@article{2022_EGF_unsupervisedclover,
  title={Utilizing unsupervised learning to improve sward content prediction and herbage mass estimation},
  author={Albert, Paul and Saadeldin, Mohamed and Narayanan, Badri and Mac Namee, Brian and Hennessy, Deirdre and O'Connor, Aisling H and O'Connor, Noel E and McGuinness, Kevin},
  journal={arXiv preprint arXiv:2204.09343},
  year={2022}
}
```

Albert, Paul, et al. "Utilizing unsupervised learning to improve sward content prediction and herbage mass estimation." arXiv preprint arXiv:2204.09343 (2022).
