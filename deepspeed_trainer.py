#!/usr/bin/env python3

import torch
import deepspeed
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DeepSpeedPlugin
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from gpt2 import *
from dataset import MyDataset2

from argparse import ArgumentParser


cfg = Config()


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.emb_d)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.emb_d))
        self.drop = nn.Dropout(cfg.emb_drop)
        self.ln_f = nn.LayerNorm(cfg.emb_d)
        self.head = nn.Linear(cfg.emb_d, cfg.vocab, bias=False)
        self.block_size = cfg.block_size
        #        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])

    #    def configure_sharded_model(self):
    #        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters())

    def forward(self, idx):
        b, t = idx.size()

        assert t <= self.block_size

        token_embeddings = self.tok_emb(idx)
        # add and drop pos embs
        x = self.drop(token_embeddings + self.pos_emb[:, :t, :])

        #        for block in self.blocks:
        #            x = deepspeed.checkpointing.checkpoint(block, x)

        x = self.blocks(x)

        x = self.ln_f(x)

        # returning the logits
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.log("training_loss", loss)
        return loss


parser = ArgumentParser()
parser.add_argument("--gpus", type=int)
parser.add_argument("--sp_model")
parser.add_argument("--data_path")
parser.add_argument("--batch_size", default=32, type=int)
args = parser.parse_args()


train_dataset = MyDataset2(args.data_path, args.sp_model, cfg.block_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0)

model = MyModel()

trainer = Trainer(
    gpus=args.gpus,
    plugins="deepspeed_stage_2_offload",
    precision=16,
)
trainer.fit(model, train_loader)
