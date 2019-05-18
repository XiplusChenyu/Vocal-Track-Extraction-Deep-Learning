# Vocal Track Extraction
Author: Chenyu Xi (cx2219)

## Introduction
There are four models in this project: Deep Clustering Model, Hybrid Deep Clustering Model, U-net Model and UH-net Model. Models are trained on DSD100 dataset. The project is based on PyTorch.

## Scripts
- Data preprocess:
  - <code>Build_Dataset.ipynb</code>: generate dataset from DSD100
  - <code>config.py</code>: define project-level parameters
  - <code>data_loader.py</code>: define torch loader
  - <code>mel_dealer.py</code>: convert music file to melspectrogram and convert spectrogram back

- Model defination:
  - <code>unet_model.py</code>: define U-net Model and UH-net Model
  - <code>cluster_model.py</code>: define Deep Clustering Model
  - <code>hybrid_model.py</code>: define Hybrid Deep Clustering Model

- Model training:
  - <code>utils.py</code>: define loss functions
  - <code>unet_train.py</code>: train functions for u-net / uh-net model
  - <code>hd_train.py</code>: train functions for hybrid deep clustering model
  - <code>dc_train.py</code>: train functions for deep clustering model
  - <code>train_dc.ipynb</code>, <code>train_hybrid.ipynb</code> and <code>train_unet.ipynb</code>: train models
  
- Model evaluation:
  - <code>evaluation.py</code>: define evaluation functions
  - <code>music_decoder.py</code>: retrieve audio file from model outputs
  
## Sample outputs
<a href=https://drive.google.com/file/d/1DQ9MeJFN8QEesQyPz-pJY7Kl58qlDQE2/view>Original Music</a> <= <a href=https://drive.google.com/file/d/1wyLOb22Vg6qpMhmi6AqQYDf4UcaIQtBs/view> Vocal Track</a><br>
==><a href=https://drive.google.com/file/d/1fNsiGOwnoctnAHyTXx5Jc5Gvqz9gSBJ9/view> Hybrid Deep Clustering Model </a><br>
==><a href=https://drive.google.com/file/d/1Ck8FNQrPbp0hc5mZ_rGQ8SXnduj6KngG/view> U-net Model </a><br>
==><a href=https://drive.google.com/file/d/1cf57rvIa7g6nA5OTiYfUdu-vEJeLLySs/view> UH-net Model </a><br>
