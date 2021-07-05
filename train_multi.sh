CUDA_VISIBLE_DEVICES=1,2,3,9 python -m torch.distributed.launch --nproc_per_node 4  --master_port=2341  semantic_main.py
