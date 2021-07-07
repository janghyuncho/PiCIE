EVAL_PATH='../../picie.pkl' # Your checkpoint directory. 

mkdir -p results/picie/test/

python train_picie.py \
--data_root datasets/coco/ \
--eval_only \
--save_root results/picie/test/ \
--eval_path ${EVAL_PATH} \
--stuff --thing  \
--res 320 
