pip install -r r.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
pip install h5py
python train.py deep_lesion
# python metric.py
# CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 DD_Net.py
