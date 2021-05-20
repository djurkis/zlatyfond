#!/usr/bin/env python3

import sentencepiece as spm


from torch.utils.data import Dataset
import numpy as np
import torch
from toolz.itertoolz import concat

from functools import partial
from multiprocessing import Pool 


class MyDataset2(Dataset):
    def __init__(self,path,model,context_len=128,sp_workers=4):
        with open(path,'r') as f:
            self.data = f.read()
        self.sp = spm.SentencePieceProcessor(model_file=model)
        
        encode = partial(self.sp.encode,out_type=int)
        # TODO: fix memory inefficiency for large textfiles
        with multiprocessing.Pool(sp_workers) as p:
            codes = p.imap(encode,self.data.split('\n'))
        self.encoded = np.array(codes)    
        self.context_len = context_len
    
    def __len__(self):
        return len(self.encoded) - self.context_len -1

    # returns context and label
    def __getitem__(self,i):
        x = torch.tensor(self.encoded[i:i+self.context_len], dtype=torch.long )
        y = torch.tensor(self.encoded[i+self.context_len+1], dtype=torch.long )
        return x,y



def main():
    pass



if __name__ == '__main__':
    main()
