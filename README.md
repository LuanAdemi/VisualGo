# VisualGo
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LuanAdemi/VisualGo/HEAD)
<a href="https://www.kaggle.com/luanademi/visualgo"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
[![forthebadge](https://forthebadge.com/images/badges/works-on-my-machine.svg)](https://forthebadge.com)

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

# Dataset
<img src="https://raw.githubusercontent.com/LuanAdemi/VisualGo/41d58899ea0b458bb8acf5587ef07286c83b8993/assets/blenderScript.png">
<img src="https://raw.githubusercontent.com/LuanAdemi/VisualGo/main/assets/header2.png">

As already mentioned, the images are actually photorealistic renders of random go boards with randomized materials, camera positions and lighting (a deeper insight on how the data was generated is given in the EDA notebook). 

You can find the dataset in its final form on <a href="https://www.kaggle.com/luanademi/visualgo">kaggle</a>. 

It was rendered using blender and the following <a href="https://gist.github.com/LuanAdemi/6aac83f06d8d4394abc22e450af18a41">script</a>.



# Inspiration
- The awesome <a href="https://www.alphagomovie.com/">AlphaGo Movie</a>
- Some other repositories (please check them out):
  - https://github.com/maciejczyzewski/neural-chessboard [![arXiv](https://img.shields.io/badge/arXiv-1708.03898-b31b1b.svg)](https://arxiv.org/abs/1708.03898)
  - https://github.com/pmauchle/ChessVision <-- this one actually uses almost the same approach :D

# References
- [1] **U-Net: Convolutional Networks for Biomedical Image Segmentation**, *Olaf Ronneberger, Philipp Fischer, Thomas Brox*  
[![arXiv](https://img.shields.io/badge/arXiv-1505.04597-b31b1b.svg)](https://arxiv.org/abs/1505.04597)
- [2] **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**, *Ramprasaath R. Selvaraju, Michael Cogswell, et al.*
[![arXiv](https://img.shields.io/badge/arXiv-1610.02391-b31b1b.svg)](https://arxiv.org/abs/1610.02391)

# Techstack
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

