![](https://img.shields.io/badge/python-3.6-brightgreen.svg) ![](https://img.shields.io/badge/pytorch-1.2.0-orange.svg)

# A Deep Learning System for Predicting Size and Fit in Fashion E-Commerce

An (unofficial) implementation of SizeFitNet (SFNet) architecture proposed in the paper [A Deep Learning System for Predicting Size and Fit in Fashion E-Commerce](https://arxiv.org/pdf/1907.09844.pdf) by Sheikh et. al (RecSys'19).

## Model Architecture

## Dataset

## Requirements

## Instructions

## Primary Results
### Learning Curves

### Performance on Test Set

## TODO
Some future work for this repository and ideas are plausible ways to improve the results:
* L2-regularization on the embeddings
* Batch Normalization on the feed forward layers
* Early Stopping
* Learning Rate Decay
* Weighted Loss Function to account for the class-imbalance
* Contrastive Learning to encourage learning different sub-spaces for positive and negative size-fits

## Acknowledgements
Thanks to Rishab Mishra for making the datasets used here publicly available on [Kaggle](https://www.kaggle.com/rmisra/clothing-fit-dataset-for-size-recommendation). Some ideas for pre-processing the data were borrowed from [NeverInAsh](https://github.com/NeverInAsh/fit-recommendation).

