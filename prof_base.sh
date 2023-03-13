
# $1 gpuid
# $2 runid

# base method
data=pacs #(mnist/cifar10/pacs)
backbone=regnet_large #regnet_large, regnet 
pretrained=True
projection_dim=2048 #resnet18:512, alexnet=4096
epochs=5 #default:5/200
batchsize=32 #16

lr=1e-4 # for pacs 1e-4
lr_scheduler=step #none/cosine/step
optimizer=sgd #adam/sgd -adam best
# Base Model Path
svroot=saved-model/prof/${data}/base_${backbone}_${pretrained}_${projection_dim}_run${2} 

python3 main_base.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr ${lr} --batchsize ${batchsize} --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim} --lr_scheduler ${lr_scheduler} --optimizer ${optimizer}
python3 main_test.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


