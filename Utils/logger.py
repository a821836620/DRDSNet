import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
from tensorboardX import SummaryWriter
class Train_logger():
    def __init__(self, save_dir, save_name):
        self.log = None
        self.summary = None
        self.save_path = save_dir
        self.save_name = save_name

    def save_model_graph(self, model, size=(1,1,64,128,128)):
        if self.summary is None:
            self.summary = SummaryWriter('%s/log' % self.save_path)
        self.summary.add_graph(model, torch.randn(size).cuda())

    def update(self, epoch, train_log, val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        print("\033[0;33mTrain:\033[0m", train_log)
        print("\033[0;33mValid:\033[0m", val_log)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self, item):
        tmp = pd.DataFrame(item, index =[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' %(self.save_path, self.save_name), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/log' % self.save_path)
        epoch = item['epoch']
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)

class Test_logger():
    def __init__(self,save_dir,save_name):
        self.log = None
        self.summary = None
        self.save_path = save_dir
        self.save_name = save_name

    def update(self,log):
        #item = OrderedDict({'img_name':name})
        #item.update(log)
        print("\033[0;33mTest:\033[0m",log)
        self.update_csv(log)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' %(self.save_path,self.save_name), index=False)

def save_args(save_path, args):
    with open(os.path.join(save_path,'args.txt'),'w') as f:
        f.writelines('----------save argparse------------\n')
        for arg, val in sorted(vars(args).items()):
            print('%s : %s \n'%(str(arg),str(val)))
            f.writelines('%s : %s \n'%(str(arg),str(val)))
        f.writelines('------------end------------------')