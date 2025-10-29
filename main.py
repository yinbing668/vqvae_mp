import argparse

import numpy as np
import torch
from torch import nn
from torchvision import datasets,transforms

# full_path = os.path.join(os.path.realpath("../../continual"), "split_data")
# sys.path.append(full_path)
# from OriLoader import OriLoader

from vqvae import VqVae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mean=(0.1,) # Mean and std including the padding
std=(0.2752,)
dat={}
dat['train']=datasets.MNIST('../gsparsity/data/Five_data/',train=True,download=True,transform=transforms.Compose([
    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))
dat['test']=datasets.MNIST('../gsparsity/data/Five_data/',train=False,download=True,transform=transforms.Compose([
    transforms.Pad(padding=2,fill=0),transforms.ToTensor(),transforms.Normalize(mean,std)]))

from torch.utils.data import DataLoader

def main(args):
    # ori_loader = OriLoader("mnist", args.batch_size)
    data_loader = DataLoader(dat['train'], shuffle=True, 
                             pin_memory=True, 
                             batch_size=args.batch_size,
                             num_workers=20)
    vae = VqVae(data_loader, device, args)
    print("*" * 10)
    print(vae.encoder)
    print("*" * 10)
    print(vae.decoder)
    vae.train_vqvae()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("argument for training")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--num_codes", type=int, default=50, help="num_latents")
    parser.add_argument("--code_dim", type=int, default=8, help="latent_dims")
    parser.add_argument("--num_latents", type=int, default=100, help="num_latents")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--epoches", type=int, default=128, help="epoches")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    main(args)
