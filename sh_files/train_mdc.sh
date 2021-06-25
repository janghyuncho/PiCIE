K_train=27
K_test=27
bsize=256
num_epoch=10
res=320
KM_INIT=20
KM_NUM=1
KM_ITER=10
SEED=2021

mkdir -p results/mdc/${KM_INIT}_${KM_NUM}_${KM_ITER}_${SEED}/ 

python train_mdc.py \
--data_root datasets/coco/ \
--save_root results/mdc/${KM_INIT}_${KM_NUM}_${KM_ITER}_${SEED}/ \
--pretrain \
--repeats 1 \
--seed ${SEED} \
--num_init_batches ${KM_INIT} \
--num_batches ${KM_NUM} \
--kmeans_n_iter ${KM_ITER} \
--K_train ${K_train} --K_test ${K_test} \
--stuff --thing  \
--batch_size_cluster ${bsize}  \
--num_epoch ${num_epoch} \
--res ${res} --res1 320 --res2 640 \
--augment --jitter --blur --grey --equiv --random_crop --h_flip 
