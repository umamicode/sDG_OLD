# PDEN

The code for **Causal Matching for Single Domain Generalization**.

### Reference Code
- Reference: https://github.com/lileicv/PDEN 

### Dataset

- Download the dataset from [Google drive](https://drive.google.com/drive/folders/15eOb1x0ewlIYuQtnjqVD4h4AWVyw-GRq?usp=sharing)
- Place the dataset files in the path `./data/`


# Train
- First, run the command `sh run_base.sh 0 0`. The command will train the base network.
- Then, run the command `sh run_my.sh 0 0`. The command will train the sDG model.
- FYI: sh run_base.sh a b; a= gpu_id, b= run_id

# Experiment 
## Digit Experiment
- Digit Experiment: MNIST -> MNIST,MNIST-M,USPS,SVHN,SYNDIGIT
- Available Models: Custom MnistNet, Resnet18, Resnet50

## Image Experiment
- Image Experiment: CIFAR10 -> CIFAR10,CIFAR10-c,CIFAR10_1
- Available Models: Resnet18, Resnet50

## PACS Experiment 
- [WORK IN PROGRESS]
- PACS Experiment: P -> A,C,S
- Available Models: Resnet18, Resnet50

# Environment
- All environment is saved in sdg.yaml.
- Code is adjusted for use in python 3.9

