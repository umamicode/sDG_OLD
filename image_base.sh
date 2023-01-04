
# $1 gpuid
# $2 runid

# base method
data=cifar10 #(mnist/cifar10/pacs)
backbone=cifar_net #(custom/cifar_net/resnet18/resnet50/wideresnet) #mnist-custom/resnet #cifar10/pacs- resnet
pretrained=False
projection_dim=128 #(custom default:128/ resnet default: 128)
epochs=200 #default:50

lr=2e-1 #for resnet18(not trainable afterwards): 1e-2/ for mnist_net: 1e-4 #og setting = 1e-4
lr_scheduler=cosine #none/cosine
optimizer=sgd #adam/sgd
# Base Model Path
svroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run${2} 

python3 main_base.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr ${lr} --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim} --lr_scheduler ${lr_scheduler} --optimizer ${optimizer}
python3 main_test.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


