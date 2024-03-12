# load ckpt at "/home/xiyu/projects/AR-LDM/ckpts/flintstones_train_unseen/epoch=99-slaghoople.ckpt"
import torch

ckpt_path = "/home/xiyu/projects/AR-LDM/ckpts/flintstones_train_10_unseen_256_distill/epoch=9-slaghoople.ckpt"
ckpt = torch.load(ckpt_path)
state_dict = ckpt['state_dict']
# print with indent
for k, v in state_dict.items():
    print(f"{k}")