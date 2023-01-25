
# $1 gpuid
# $2 runid

# cifar10 method
w_cls=1.0
w_cyc=20
w_info=0.1 
w_oracle=1.0
w_div=2.0 
div_thresh=0.5
w_tgt=1.0
n_tgt=20
max_tgt=19
tgt_epochs_fixg=15 #15
tgt_epochs=30 #30
lmda=0.051 #lmda for adv-barlowtwins (0.051 best)

gen=cnn
interpolation=img
oracle=False


data=cifar10 #mnist/cifar10/pacs
backbone=cifar_net #(cifar_net/resnet18/resnet50/wideresnet) 
pretrained=False #Only to load right base model. my_iter process is set as pretrained=False.
projection_dim=256 #default: (mnist: 128/ cifar-10:128)
loss_fn=barlowtwins #supcon/barlowtwins/barlowquads/prism/vicreg
lr=1e-4 #1e-4 #1e-3 sucks #adam with 1e-5/1e-4 #sdg with 1e-4
lr_scheduler=none #cosine/none  #adam with none # sgd with cosine
optimizer=adam #sgd/adam

batchsize=128 #default:128 
c_level=1

# Model Load/Save Path
baseroot=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run0/best.pkl
svpath=saved-model/${data}/base_${backbone}_${pretrained}_${projection_dim}_run0
# step1
python3 main_test.py --gpu $1 --modelpath ${baseroot} --svpath ${svpath}/test_${c_level}.log --backbone ${backbone} --projection_dim ${projection_dim} --data ${data} --c_level ${c_level}

#done
