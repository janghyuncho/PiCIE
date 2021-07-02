EVAL_PATH='../../picie.pkl' # Your checkpoint directory. 

mkdir -p results/test/picie/

python train_picie.py \
--data_root datasets/coco/ \
--eval_only \
--save_root results/test/picie \
--eval_path ${EVAL_PATH} \
--stuff --thing  \
--res 320 
