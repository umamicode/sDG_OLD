
# $1 gpuid
# $2 runid

# base method
data=mnist #(mnist/cifar10)
backbone=resnet50
pretrained=False
epochs=50 #default:50

# Base Model Path
svroot=saved-model/${data}/base_${backbone}_${pretrained}_run${2} 

python3 main_base.py --gpu $1 --data ${data} --epochs ${epochs} --nbatch 100 --lr 1e-4 --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} 
python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} 


