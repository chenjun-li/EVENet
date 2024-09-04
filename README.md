# EVENet
"EVENet: Evidence-based Ensemble Learning for Uncertainty-aware Brain Parcellation using Diffusion MRI"
## Overview
Evidence-based Ensemble Neural Network (EVENet) is a novel uncertainty-aware deep learning method for anatomical brain parcellation of cortical and subcortical regions directly from dMRI data. The key innovation of EVENet is its utilization of evidential deep learning to quantify uncertainty at each voxel during a single inference. The development of the model is based on [FastSurfer](https://github.com/Deep-MI/FastSurfer/tree/dev).
## Usage
### Installation
EVENet is built and tested in an environment the same as that specified by FastSurfer (See [here](https://github.com/Deep-MI/FastSurfer/blob/dev/doc/overview/INSTALL.md#native-ubuntu-2004-or-ubuntu-2204) for more details). For a native install on Ubuntu 22.04, simply clone the project and create a new conda environment and run `conda env create -f evenet_cpu.yml` or `conda env create -f evenet_gpu.yml`.

### Running EVENet Parcellation

The input of the pretrained EVENnet models must be **320x320x320** dMRI image with a voxel size of **1.25x1.25x1.25 \( mm^3 \)**.

The `script_to_run.txt` file provides the example command lines to run EVENet. For example, to obtain the anatomical brain parcellation on a `FA` image of `Subject X`, run
```


```
### Perform Ensemble
