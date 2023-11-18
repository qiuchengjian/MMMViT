#!/bin/bash
#pythonname='pytorch_1.2.0a0+8554416-py36tf'

dataname='BRATS2018'
#pypath=$pythonname
#cudapath=cuda-9.0
datapath='your datapath'
savepath='your model savepath'
 
#export CUDA_VISIBLE_DEVICES=0,1,2,3

#export PATH=$cudapath/bin:$PATH
#export LD_LIBRARY_PATH=$cudapath/lib64:$LD_LIBRARY_PATH
#PYTHON=$pypath/bin/python3.6
#export PATH=$pypath/include:$pypath/bin:$PATH
#export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

python train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 1000 --dataname $dataname

#eval:
#resume='/media/image522/221D883AB7DA6CE4/qcj/mmFormer/DB/model_path/model_last.pth'
#python train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 0 --dataname $dataname --resume $resume

#师弟，我这个修改论文比较急，一张显卡同事跑多个实验比较慢，我就用两天


