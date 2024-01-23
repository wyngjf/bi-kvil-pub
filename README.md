# Keypoint-based Visual Imitation Learning (K-VIL)

---

This repository contains source code for the following papers:

1. Keypoint-based Visual Imitation Learning (K-VIL), [Project Website](https://sites.google.com/view/k-vil) 
   1. uni-manual
   2. single layer of master-slave relationship between two objects
   
2. Keypoints-based Visual Imitation Learning of Bimanual Manipulation Tasks (Bi-KVIL) [Project Website](https://sites.google.com/view/bi-kvil) 
   1. bimanual manipulation


---
## Installation

If conda is not yet available on your computer, install [Miniconda, e.g., for python 3.10](https://docs.conda.io/en/latest/miniconda.html#linux-installers). To run the downloaded script, you may need to enable its execution using `chmod u+x ~/Downloads/Miniconda[...].sh`. After the installation, you can disable the auto-activation of the default "base" environment, as described in the installer output.

Afterwards, get and run the install script for kvil. If you modify the paths, be aware to either add or leave out the concluding slash (`/`) dependent on how the default path is defined.

```shell
wget https://raw.githubusercontent.com/wyngjf/bi-kvil-pub/main/kvil_install.sh
source kvil_install.sh
```

**dependencies**

These and other dependencies are automatically added by the install script mentioned beforehand:

- [robot-utils](https://gitlab.com/jianfenggaobit/robot-utils) package: this package includes utilities for robotics 
  research in python..
- [robot-vision](https://gitlab.com:jianfenggaobit/robot-vision) package: contains computer vision models suited for robotic tasks.


## Instructions:

see the individual tutorials for each paper:
1. [K-VIL](./docs/kvil.md)
2. [K-VIL 2.0](./docs/kvil2.md)