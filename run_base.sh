
# $1 gpuid
# $2 runid

# base method
data=cifar10 #(mnist/cifar10/pacs)
backbone=custom #(custom/resnet18/resnet50) #mnist-custom/resnet #cifar10/pacs- resnet
pretrained=False
projection_dim=258 #Higher Dim. for BarlowTwins (custom default:128/ resnet default: 128)
epochs=100 #default:50

# Base Model Path
svroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run${2} 

python3 main_base.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr 1e-4 --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim}
python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


