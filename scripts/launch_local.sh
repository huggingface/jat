torchrun \
--nnodes=1 \
--nproc_per_node 4 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint 127.0.0.1:29500 \
gia/train.py