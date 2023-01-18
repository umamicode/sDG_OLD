# PDEN

The code for **Causal Matching for Single Domain Generalization**.

### Reference Code
- Reference: https://github.com/lileicv/PDEN 

### Dataset

- Download the dataset from [Google drive](https://drive.google.com/drive/folders/14F78O0IAZ1zc8-tKOVbENYYtADkazuJq?usp=sharing)
- Place the dataset files in the path `./data/`


# Train
- First, run the command `sh {$data}_base.sh 0 0`. The command will train the base network.
- Then, run the command `sh {$data}.sh 0 0`. The command will train the sDG model.
- FYI: sh run_base.sh a b; a= gpu_id, b= run_id

# Experiment 
## Digit Experiment
- Digit Experiment: MNIST -> MNIST,MNIST-M,USPS,SVHN,SYNDIGIT
- Available Models: Custom MnistNet, Resnet18, Resnet50

## CIFAR10-C Experiment
- Image Experiment: CIFAR10 -> CIFAR10,CIFAR10-c,CIFAR10_1
- Available Models: Resnet18, Resnet50, WideResnet

## PACS Experiment 
- PACS Experiment: P -> A,C,S
- Available Models: Resnet18, Resnet50, WideResnet

# Environment
- All environment is saved in sdg.yaml.
- Code is adjusted for use in python 3.9

