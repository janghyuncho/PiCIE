K_train=27
K_test=27
bsize=256
num_epoch=10
KM_INIT=20
KM_NUM=1
KM_ITER=20
SEED=1
LR=1e-4

mkdir -p results/picie/train/${SEED}

python train_picie.py \
--data_root datasets/coco/ \
--save_root results/picie/train/${SEED} \
--pretrain \
--repeats 1 \
--lr ${LR} \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--stuff --thing  \
--batch_size_cluster ${bsize} \
--num_epoch ${num_epoch} \
--res 320 --res1 320 --res2 640 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip 
