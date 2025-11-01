import shutil
# from numpy import square
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import MnistEncoder, Decoder, Coding
# from torchvision.datasets import mnist
import os

model_dict={
    'fashion_mnist':()
}

class VqVae(nn.Module):
    
    
    def __init__(self, dataloader, device, args):
        """
        vqvae:learning
        """
        super().__init__()
        # exports the hyperparameters to a yaml file, and create "self.hparams" namespace
        # create model
        self.args = args
        self.kld_reg = args.kld_reg
        self.num_codes = args.num_codes
        self.code_dim = args.code_dim
        self.dataloader = dataloader
        self.device = device
        self._create_net(device)

        shutil.rmtree("runs")
        self.writer = SummaryWriter("runs/vqvae")

    def _create_net(self, device):
        
        self.encoder = MnistEncoder().to(device)
        self.decoder = Decoder().to(
            device
        )
        self.coding = Coding(self.num_codes, self.code_dim).to(device)

   
    def step(self, x):
        enc_oup = self.encoder(x)
        code, new_representation = self.coding(enc_oup)
        new_repre_detach = enc_oup+(new_representation-enc_oup).clone().detach()
        dec_oup = self.decoder(new_repre_detach)
        return enc_oup, code, new_representation, dec_oup

    def generate_images(self, codes, emb_shape):
        embedding = self.coding.embedding_code(codes, emb_shape)
        images = self.decoder(embedding)
        return images

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
            loss_original = (x - dec_oup).pow(2).mean()

            loss_representation = (enc_oup - new_representation).pow(2).mean()
            optimizer.zero_grad()
            loss = loss_original + self.kld_reg*loss_representation
            loss.backward()
            optimizer.step()
            if batch == 0:
                self.print_grad()
                print(code.shape)
                print(list(code[:10].cpu().numpy()))
            if batch % 50 == 0:
                print(
                    f"""Epoch{epoch}, batch{batch}, recon loss:{loss_original.item():.4f},
                    coding loss:{loss_representation.item():.4f}, 
                    total loss:{loss.item():.4f}""",
                )
        self.writer.add_scalar('loss/images', loss_original.item(), epoch)
        self.writer.add_scalar('loss/representation', loss_representation.item(), epoch)
        self.writer.add_scalar('loss/total', loss.item(), epoch)

    def train_vqvae(self):
        for epoch in range(self.args.epoches):
            self.train_loop(epoch)
            self.sample(epoch)
            if epoch%5==0:
                self.save_checkpoint()

    def save_checkpoint(self):
        result_dir = './result'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        data_dict = {
            "vocab_size":self.num_codes, 
            "code_dim":self.code_dim,
            "data":self.get_codes()
            }
        torch.save(data_dict, './result/codes.pt')
        torch.save(self.state_dict(), './result/vqvae.pt')

    @classmethod
    def from_pretrained(cls, device):
        codes_all = torch.load('./result/codes.pt')
        class Args:
            kld_reg = None
            num_codes = codes_all["vocab_size"]
            code_dim = codes_all["code_dim"]
        vqvae = VqVae(None, device, Args())
        state_dict = torch.load('./result/vqvae.pt')
        # print(state_dict)
        vqvae.load_state_dict(state_dict)
        return vqvae


    @staticmethod
    def resize_to_image(x):
        return torch.unflatten(x, 1, (1, 28, 28))

    def sample(self, i):
        x, y = next(iter(self.dataloader))
        x = x.to(self.device)
        enc_oup, code, new_representation, dec_oup = self.step(x)
        self.writer.add_images(f"{i}original_samples", x, 0)
        self.writer.add_images(f"{i}generation", dec_oup, 0)

    def get_codes(self):
        codes_all = []
        for batch, (x, _) in enumerate(self.dataloader):
            # print(x.shape)
            x = x.to(self.device)
            with torch.no_grad():
                enc_oup, code, new_representation, dec_oup = self.step(x)
            codes_all.append(code.cpu())
        codes_all = torch.concatenate(codes_all)
        return codes_all

# import torch

# # Create a tensor
# t = torch.tensor([1.0, 2.0, 3.0])

# # Save the tensor to a file
# torch.save(t, 'my_tensor.pt')

# # Create a dictionary of tensors
# data_dict = {'a': torch.tensor([10, 20]), 'b': torch.tensor([30, 40])}

# # Save the dictionary of tensors
# torch.save(data_dict, 'tensor_data.pt')