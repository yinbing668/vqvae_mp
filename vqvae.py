import shutil
# from numpy import square
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Decoder, Coding
# from torchvision.datasets import mnist


class VqVae(nn.Module):
    def __init__(self, dataloader, device, args):
        """
        vqvae:learning
        """
        super().__init__()
        # exports the hyperparameters to a yaml file, and create "self.hparams" namespace
        # create model
        self.args = args
        self.num_latents = args.num_latents
        self.code_dim = args.code_dim
        self.num_codes = args.num_codes
        self.dataloader = dataloader
        self.device = device
        self._create_net(device)
        shutil.rmtree("runs")
        self.writer = SummaryWriter("runs/vqvae")

    def _create_net(self, device):
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(
            device
        )
        self.re_represent = Coding(100, 16).to(device)

    def create_embed(self, code):
        oup = self.decoder.embbing(code)
        oup = torch.flatten(oup, start_dim=1)
        return oup

   
    def step(self, x):
        enc_oup = self.encoder(x)
        code, new_representation = self.re_represent(enc_oup)
        new_repre_detach = enc_oup+(new_representation-enc_oup).clone().detach()
        dec_oup = self.decoder(new_repre_detach)
        return enc_oup, code, new_representation, dec_oup

    def print_grad(self):
        enc_grad = [para.grad.abs().mean() for para in self.encoder.parameters()]
        dec_grad = [para.grad.abs().mean() for para in self.decoder.parameters()]
        print("enc grad", enc_grad)
        print("dec grad", dec_grad)

    def train_loop(self, epoch):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.95)
        for batch, (x, _) in enumerate(self.dataloader):
            # print(x.shape)
            x = x.to(self.device)
            enc_oup, code, new_representation, dec_oup = self.step(x)
            loss_original = (x - dec_oup).square().mean()
            loss_representation = (enc_oup - new_representation).square().mean()
            optimizer.zero_grad()
            loss = loss_original + 0.01*loss_representation
            loss.backward()
            optimizer.step()
            if batch == 0:
                self.print_grad()
                print(code.shape)
                print(list(code[:10].cpu().numpy()))
            if batch % 50 == 0:
                print(
                    f"Epoch{epoch}, batch{batch},dec loss:{loss.item():.4f}",
                )
        self.writer.add_scalar('loss/images', loss_original.item(), epoch)
        self.writer.add_scalar('loss/representation', loss_representation.item(), epoch)

    def train_vqvae(self):
        for epoch in range(self.args.epoches):
            self.train_loop(epoch)
            self.sample(epoch)

    @staticmethod
    def resize_to_image(x):
        return torch.unflatten(x, 1, (1, 28, 28))

    def sample(self, i):
        x, y = next(iter(self.dataloader))
        x = x.to(self.device)
        enc_oup, code, new_representation, dec_oup = self.step(x)
        self.writer.add_images(f"{i}original_samples", x, 0)
        self.writer.add_images(f"{i}generation", dec_oup, 0)
