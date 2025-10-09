from tkinter import NO
from turtle import forward
from numpy import pad
from torch import nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, padding=1),#14
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, padding=1),#7
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, padding=1),
        )
        # self.Flatten = nn.Flatten()

    def forward(self, x):
        # out = self.Flatten(x)
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embbing = nn.Embedding(num_codes, code_dim, max_norm=1.0)
        # self.embbing.weight.data.uniform_(-1.0 / code_dim, 1.0 / code_dim)
        self.net = nn.Sequential(
            # start shape (b, 16, 7, 7)
            nn.ConvTranspose2d(16, 16, 3, 2, padding=1, output_padding=1),#expected 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 3, 1, padding=1),#expected 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 1, padding=1),
        )
        
    def forward(self, x):
        # x_emb = self.embbing(x)
        # print(x_emb.shape)
        # x_flat = torch.flatten(x_emb, start_dim=1)
        return self.net(x)

class Coding(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        weight = torch.randn((num_codes, code_dim))
        weight = F.normalize(weight, dim=-1)
        self.embedding = nn.Embedding(num_codes, code_dim).from_pretrained(weight)
        # self.embedding.weight.data.uniform_(-1.0 /code_dim, 1/code_dim)

        
    def forward(self, x):
        # x shape(b, c, h, w)
        inp_shape = x.shape
        x = torch.flatten(x, start_dim=2)
        # x shape(b, c, h*w)
        x = torch.permute(x, (0, 2, 1))[:,:,None,:]
        # x = torch.nn.functional.normalize(x, p=2., dim=-1)
        # x shape(b, h*w, 1, c)
        oup_emb_diff = x - self.embedding.weight
        oup_emb_distance = torch.norm(oup_emb_diff, dim=-1)
        code = torch.argmin(oup_emb_distance, dim=-1)
        # code shape (b, h*w)
        x_new = self.embedding(code)
        # x_new shape(b, h*w, c)
        x_new = torch.permute(x_new, (0, 2, 1))
        x_new = torch.reshape(x_new, inp_shape)
        return code, x_new
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.rand(10, 1, 28, 28).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(
        device
    )
    re_represent = Coding(100, 16).to(device)
    enc_oup = encoder(x)
    print('enc oup shape:', enc_oup.shape)
    code, new_representation = re_represent(enc_oup)
    print('new represent shape:', new_representation.shape)
    dec_oup = decoder(new_representation)
    print('-'*8, 'shape after decode')
    print(dec_oup.shape)