EVAL_PATH='picie.pth.tar' # Your checkpoint directory. 

python train_picie.py \
--data_root datasets/coco/ \
--eval_only \
--eval_path ${EVAL_PATH} \
--stuff --thing  \
--res 320 
