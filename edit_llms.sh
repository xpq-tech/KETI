# CUDA_VISIBLE_DEVICES=1 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/llama3.1-8b > logs/ft_llama3.1-8b.log &


# CUDA_VISIBLE_DEVICES=0 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/llama3.1-8b > logs/grace_llama3.1-8b.log &

# CUDA_VISIBLE_DEVICES=1 nohup python edit_main.py \
#     --editing_method=UNKE \
#     --hparams_dir=./hparams/UNKE/llama3.1-8b > logs/unke_llama3.1-8b.log &

# CUDA_VISIBLE_DEVICES=6 nohup python edit_main.py \
#     --editing_method=FT \
#     --hparams_dir=./hparams/FT/llama2-13b > logs/ft_llama2-13b-2.log &

# CUDA_VISIBLE_DEVICES=2 nohup python edit_main.py \
#     --editing_method=GRACE \
#     --hparams_dir=./hparams/GRACE/llama2-13b > logs/grace_llama2-13b.log &

# CUDA_VISIBLE_DEVICES=4,5,6 nohup python edit_main.py \
#     --editing_method=UNKE \
#     --hparams_dir=./hparams/UNKE/llama2-13b > logs/unke_llama2-13b.log &



### Non Edit
# CUDA_VISIBLE_DEVICES=1 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/llama3.1-8b > logs/non_edit_llama3.1-8b.log &

# CUDA_VISIBLE_DEVICES=4 nohup python edit_main.py \
#     --editing_method=non-edit \
#     --hparams_dir=./hparams/NON_EDIT/llama2-13b > logs/non_edit_llama2-13b.log &