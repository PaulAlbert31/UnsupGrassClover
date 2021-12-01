CUDA_VISIBLE_DEVICES=0 python main_iMix.py --net resnet18 --dataset irish --epochs 2000 --steps 1000 1500 --lr 0.01 --batch-size 128 --save-dir irish_resnet18_iMix
CUDA_VISIBLE_DEVICES=0 python main_iMix.py --net resnet18 --dataset danish --epochs 2000 --steps 1000 1500 --lr 0.01 --batch-size 128 --save-dir danish_resnet18_iMix

CUDA_VISIBLE_DEVICES=0 python main.py --net resnet18 --dataset irish --epochs 100 --steps 50 80 --batch-size 16 --save-dir clover_res18_irish --seed 1 --lr 0.001 --pretrained irish_resnet18_iMix/last_model.pth.tar --red
CUDA_VISIBLE_DEVICES=0 python main.py --net resnet18 --dataset danish --epochs 100 --steps 50 80 --batch-size 16 --save-dir clover_res18_danish --seed 1 --lr 0.001 --pretrained danish_resnet18_iMix/last_model.pth.tar
