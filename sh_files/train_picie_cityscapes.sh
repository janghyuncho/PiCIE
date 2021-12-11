K_train=27
K_test=27
bsize=256
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=0.001

mkdir -p results/cityscapes/picie/train/${SEED}/

python train_picie.py \
--data_root datasets/cityscapes/ \
--save_root results/cityscapes/picie/train/${SEED}/ \
--pretrain \
--lr ${LR} \
--repeats 1 \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--batch_size_cluster ${bsize} \
--num_epoch ${num_epoch} \
--cityscapes \
--label_mode 'gtFine' \
--res 320 --res1 320 --res2 640 \
--augment \
--grey \
--blur \
--jitter \
--equiv \
--random_crop \
--h_flip 
