
# $1 gpuid
# $2 runid

# digit method
w_cls=1.0
w_cyc=20
w_info=0.1 
w_oracle=1.0
w_div=2.0 
div_thresh=2.5 #0.5
w_tgt=1.0

n_tgt=100 #20
max_tgt=99 #19
tgt_epochs_fixg=5 #15
tgt_epochs=10 #30
lmda=0.051 #lmda for adv-barlowtwins (0.051 best)
gen=cnn #cnn/stn
interpolation=img

oracle=False
oracle_type=ft

data=mnist #mnist/cifar10/pacs
backbone=custom #(custom)
pretrained=False #Only to load right base model. my_iter process is set as pretrained=False.
projection_dim=128 #default: (mnist: 128)
loss_fn=mdar #supcon/mdar/mdarv2

batchsize=128 #default:128 -> 256

# Model Load/Save Path
svroot=saved-model/uniform/${data}/${gen}_${interpolation}_${backbone}_${loss_fn}_${pretrained}_${projection_dim}_${w_cls}_${w_cyc}_${w_info}_${w_div}_${div_thresh}_${w_tgt}_lmda${lmda}_oracle${oracle}_${oracle_type}_${w_oracle}_run${2}
baseroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run0/best.pkl

# step1
python3 main_my_iter.py --gpu $1 --data ${data} --gen $gen --backbone ${backbone} --loss_fn ${loss_fn} --projection_dim ${projection_dim} --interpolation $interpolation --n_tgt ${n_tgt} --tgt_epochs ${tgt_epochs} --tgt_epochs_fixg ${tgt_epochs_fixg} --nbatch 100 --batchsize ${batchsize} --lr 1e-4 --w_cls $w_cls --w_cyc $w_cyc --w_info $w_info --w_div $w_div --w_oracle $w_oracle --div_thresh ${div_thresh} --w_tgt $w_tgt --ckpt ${baseroot} --svroot ${svroot} --pretrained ${pretrained} --oracle ${oracle} --lmda ${lmda} --oracle_type ${oracle_type}
python3 main_test.py --gpu $1 --modelpath ${svroot}/${max_tgt}-best.pkl --svpath ${svroot}/test.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data}

#done
