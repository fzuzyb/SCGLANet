# val track1
# CUDA_VISIBLE_DEVICES=1  python -m torch.distributed.launch --nproc_per_node=1 --master_port=4327 basicsr/test.py -opt ./options/test/StereoSR/val_SCGLANet-L_4x_Track1.yml --launcher pytorch
## final test track1
#CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=4327 basicsr/test.py -opt ./options/test/StereoSR/test_SCGLANet-L_4x_Track1.yml --launcher pytorch
## final test track2
#CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=4328 basicsr/test.py -opt ./options/test/StereoSR/test_SCGLAGAN-L_4x_Track2.yml --launcher pytorch
# final test track3
#CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 --master_port=4329 basicsr/test.py -opt ./options/test/StereoSR/test_SCGLANet-L_4x_Track3.yml--launcher pytorch
