# create a GPT instance
from torch.utils.tensorboard import SummaryWriter

from mingpt.model import GPT
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from vqvae import VqVae
class CodingData(Dataset):
    def __init__(self, codes, vocab_size):
        self.codes = codes
        self.vocab_size = vocab_size
        self.codes_shape = codes.shape
        self.block_size = self.codes_shape[1]
        self.start_ind = vocab_size
        
    def __len__(self):
        return self.codes_shape[0]
    
    def get_vocab_size(self):
        return self.vocab_size+1
    
    def get_block_size(self):
        return self.block_size
        
    def __getitem__(self, idx):
        code = self.codes[idx]
        code_pre = torch.concat((torch.tensor([self.start_ind]), code[:-1]))
        return code_pre, code

device = torch.device("cuda")
codes_all = torch.load("./result/codes.pt")

codes = codes_all['data']
coding_dataset = CodingData(codes,codes_all['vocab_size'])

vqvae = VqVae.from_pretrained(device)

model_config = GPT.get_default_config()
model_config.model_type = 'gpt-mini'
model_config.vocab_size = codes_all['vocab_size']+1
model_config.block_size = 8*8
model = GPT(model_config)
print(model)

# create a Trainer object
from mingpt.trainer import Trainer
train_config = Trainer.get_default_config()
train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
train_config.max_iters = 10000
train_config.num_workers = 15
trainer = Trainer(train_config, model, coding_dataset)
bsz=8
prefix = torch.LongTensor([coding_dataset.start_ind]*bsz)[:,None].to(device)
writer = SummaryWriter("runs/vqvae")
def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 99:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}",end="")
        codes_gene = model.generate(prefix,max_new_tokens=coding_dataset.block_size, do_sample=True)
        # codes_gene = torch.LongTensor(codes_gene)
        codes_gene = codes_gene.to(dtype=torch.int64)
        codes_gene = codes_gene[:,1:]
        codes_gene[codes_gene==100] = 99
        # print(codes_gene.shape)
        # print(codes_gene)
        # print(codes_gene[:,1:])
        images = vqvae.generate_images(codes_gene, (bsz,16,8,8))
        ppl = trainer.calc_ppl()
        print(f" ppl:{ppl:.3f}")
        writer.add_images(f"{trainer.iter_num}samples", images, 0)
        
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()
torch.save(model.state_dict(), 'gpt.pth')
