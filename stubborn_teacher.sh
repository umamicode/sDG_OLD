
# $1 gpuid
# $2 runid

# base method
data=pacs #(mnist/cifar10/pacs)
backbone=resnet18 #(pacs_net/resnet18/resnet50/wideresnet) 
pretrained=True
projection_dim=512 #(custom default:128/ cifar_net default: 128) default may also be 1024 -max (8192,16384)
epochs=50 #default:5/200
batchsize=16 #16

lr=1e-2 # for pacs 1e-4
lr_scheduler=none #none/cosine
optimizer=sgd #adam/sgd -adam best
# Base Model Path
svroot=saved-model/steacher/${data}/base_${backbone}_${pretrained}_${projection_dim}_run${2} 

python3 main_teacher.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr ${lr} --batchsize ${batchsize} --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim} --lr_scheduler ${lr_scheduler} --optimizer ${optimizer}
python3 main_test.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


