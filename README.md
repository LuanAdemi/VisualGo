# VisualGo
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LuanAdemi/VisualGo/HEAD)
<a href="https://www.kaggle.com/luanademi/visualgo"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>

VisualGo is a toolset of different machine learning algorithms to extract the current go board state from an image. It features two models, which first find the go board in the given image and then predict the current state. The basic pipeline of the whole model looks like the following:

<img src="https://raw.githubusercontent.com/LuanAdemi/VisualGo/main/assets/Pipeline.png">

# Models
As seen in the figure above, the model is divided into two sub-models and a handcrafted transformer, which performs a perspective warp and threasholding on the images using the predicted masks.

The Segmentation Model is a basic UNet architecture [1] trained on ~800 rendered images of go boards. They are resized to 128x128 pixels and feeded into the network in batches of 16 images. It is a basic binary segmentation problem, hence the performance of the model is pretty good.

The State Prediction Model is a residual tower trained on the transformed images from the Segmentation Model.

# Files
This repository contains a set of notebooks explaining every model in depth and analysing their perfomance using <a href="https://captum.ai/">Captum</a>.

Here is a basic table of contents:

- EDA: Exploratory Data Analysis of the VisualGo Dataset
- Segmentation: Exploring the Segmentation model architecture and investigating the model quality using Captum
- MaskTransformer: Explaining the Transformer
- Position: Exploring the Position model architecture and investigating the model quality using Captum

I highly recommend checking them out with the <a href="https://mybinder.org/v2/gh/LuanAdemi/VisualGo/HEAD">binder link</a> I set up.


# References
- [1] **U-Net: Convolutional Networks for Biomedical Image Segmentation**, *Olaf Ronneberger, Philipp Fischer, Thomas Brox* (https://arxiv.org/abs/1505.04597)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)
