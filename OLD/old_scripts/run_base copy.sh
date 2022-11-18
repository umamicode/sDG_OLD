
# $1 gpuid
# $2 runid

# base method
data=mnist #mnist/cifar10(not ready)
backbone=resnet18
pretrained=True

svroot=saved-model/${data}/base_${backbone}_${pretrained}_run${2}  #image
#svroot=saved-digit/base_${backbone}_${pretrained}_run${2}  #digit
#example svroot= saved-digit/base_run0_${backbone}/best.pkl

python3 main_base.py --gpu $1 --data ${data} --epochs 50 --nbatch 100 --lr 1e-4 --svroot $svroot --backbone ${backbone} --pretrained ${pretrained} 
python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log --backbone ${backbone} 


