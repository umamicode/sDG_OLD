
# $1 gpuid
# $2 runid

# base method
#svroot=saved-digit/base_run${2}
#python3 main_base.py --gpu $1 --data mnist --epochs 50 --nbatch 100 --lr 1e-4 --svroot $svroot
#python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log

# AutoAugment method
#for autoaug in AA RA
#do
#    svroot=saved-digit/${autoaug}_run${2}
#    python3 main_base.py --gpu $1 --data mnist --autoaug ${autoaug} --epochs 50 --nbatch 100 --lr 1e-4 --svroot $svroot
#    python3 main_test_digit.py --gpu $1 --modelpath $svroot/best.pkl --svpath $svroot/test.log
#done

# my method
w_cls=1.0
w_cyc=20
w_info=0.1
w_div=2.0
div_thresh=0.5
w_tgt=1.0
n_tgt=20
max_tgt=19
tgt_epochs=30

gen=cnn
interpolation=img


data=cifar10 #mnist/cifar10/pacs
backbone=resnet18 #custom/resnet18/resnet50
pretrained=False #Only to load right base model. my_iter process is set as pretrained=False.
projection_dim=128 #default: (mnist: 128/ cifar-10:)
loss_fn=supcon #supcon/relic/barlowtwins

batchsize=128 #default:128

# Model Load/Save Path
svroot=saved-model/${data}/${gen}_${interpolation}_${backbone}_${loss_fn}_${pretrained}_${projection_dim}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_run${2}
baseroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run0/best.pkl

# step1
python3 main_my_iter.py --gpu $1 --data ${data} --gen $gen --backbone ${backbone} --loss_fn ${loss_fn} --projection_dim ${projection_dim} --interpolation $interpolation --n_tgt ${n_tgt} --tgt_epochs ${tgt_epochs} --tgt_epochs_fixg 15 --nbatch 100 --batchsize ${batchsize} --lr 1e-4 --w_cls $w_cls --w_cyc $w_cyc --w_info $w_info --w_div $w_div --div_thresh ${div_thresh} --w_tgt $w_tgt --ckpt ${baseroot} --svroot ${svroot} 
python3 main_test_digit.py --gpu $1 --modelpath ${svroot}/${max_tgt}-best.pkl --svpath ${svroot}/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}

#done

