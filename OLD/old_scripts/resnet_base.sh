
# $1 gpuid
# $2 runid

# base method
data=cifar10 #(mnist/cifar10/pacs)
backbone=resnet50 #(cifar_net/resnet18/resnet50/wideresnet) 
pretrained=False
projection_dim=2048 #default: 128 / trying :1024 for cifar_net , 2048 for resnet50
epochs=200 #default:50
batchsize=128 #resnet50:128/256

lr=1e-2 #1e-1 #resnet50-ft-5e-2
lr_scheduler=cosine #none/cosine/step
optimizer=sgd #adam/sgd
# Base Model Path
svroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run${2} 

python3 main_base.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr ${lr} --batchsize ${batchsize} --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim} --lr_scheduler ${lr_scheduler} --optimizer ${optimizer}
python3 main_test.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


