
# $1 gpuid
# $2 runid

# base method
data=pacs #(mnist/cifar10/pacs)
backbone=wideresnet #(pacs_net/resnet18/resnet50/wideresnet) 
pretrained=True
projection_dim=512 #(custom default:128/ cifar_net default: 128) default may also be 1024 -max (8192,16384)
batchsize=16 #16

lr=1e-4 # for pacs 1e-4
lr_scheduler=none #none/cosine
optimizer=adam #adam/sgd -adam best
w_oracle=1.0
oracle_epoch=5
epochs=100 #default:5/200
autoaug=None


# Base Model Path
svroot=saved-model/prof/${data}/base_${backbone}_${pretrained}_${projection_dim}_${autoaug}_woracle${w_oracle}_epoch${epochs}_run${2} 

python3 main_teacher.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr ${lr} --batchsize ${batchsize} --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} --projection_dim ${projection_dim} --lr_scheduler ${lr_scheduler} --optimizer ${optimizer} --w_oracle ${w_oracle} --oracle_epoch ${oracle_epoch} --autoaug ${autoaug}
python3 main_test.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}


