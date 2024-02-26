# Bi-KVIL: Keypoints-based Visual Imitation Learning of Bimanual Manipulation Tasks

Official Implementation of the [Bi-KVIL paper](https://sites.google.com/view/bi-kvil)
- [Link to the website of Bi-KVIL](https://sites.google.com/view/bi-kvil)
- [Link to the previous work KVIL](https://sites.google.com/view/k-vil)

BibTeX:
```
@ARTICLE {gao_kvil_2023,
    author = {Jianfeng Gao and Zhi Tao and Noémie Jaquier and Tamim Asfour},
    title = {K-VIL: Keypoints-based Visual Imitation Learning},
    pages = {3888--3908},
    volume ={39},
    number ={5},
    journal ={IEEE Transactions on Robotics},
    year = {2023}
}
@INPROCEEDINGS {gao_bikvil_2024,
    author = {Jianfeng Gao and Xiaoshu Jin and Franziska Krebs and Noémie Jaquier and Tamim Asfour},
    title = {Bi-KVIL: Keypoints-based Visual Imitation Learning of Bimanual Manipulation Tasks},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year = {2024}
}
```

---
## Installation

If conda is not yet available on your computer, 
install [Miniconda, e.g., for python 3.10](https://docs.conda.io/en/latest/miniconda.html#linux-installers). 
To run the downloaded script, you may need to enable its execution using `chmod u+x ~/Downloads/Miniconda[...].sh`. 
After the installation, you can disable the auto-activation of the default "base" environment, as described in the installer output.

Get and run the installation script for kvil. If you modify the paths, be aware to either add or leave out the concluding slash (`/`) dependent on how the default path is defined.

```shell
wget https://raw.githubusercontent.com/wyngjf/bi-kvil-pub/main/scripts/kvil_install.sh
source kvil_install.sh
```

After installation, run

```shell
robot_vision_install
```
and select (up / down arrow to navigate and space to select or deselect) the following dependencies.
- apex
- graphormer
- groundingdino
- mmpose
- opendr
- raft
- sam
- aot
- unimatch

Pay attention to the terminals. If you don't have CUDA configured properly, you may encounter problems when installing 
`groundingDINO`. The script will just continue with error. You need to solve those issues accordingly and run the command again.

## Instructions

### Recording

see tutorial on [Recording](docs%2F_tutorial_record_demo.md)

### Preprocessing and K-VIL

see [Demo Preprocessing and K-VIL](docs%2F_totorial_demo_preprocessing.md)