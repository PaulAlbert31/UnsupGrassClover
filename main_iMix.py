import argparse
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

from utils_iMix import make_data_loader, mixup_data
import os
from torch.multiprocessing import set_sharing_strategy
set_sharing_strategy("file_system")

from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, args):
        self.args = args

        if args.net == "resnet18":
            from nets.resnet_unsup import ResNet18
            model = ResNet18(self.args.proj_size)
        else:
            raise NotImplementedError
        
        print("Number of parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        self.model = nn.DataParallel(model).cuda()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.steps, gamma=self.args.gamma)

        self.criterion = nn.CrossEntropyLoss()
        
        self.kwargs = {"num_workers": 12, "pin_memory": True, "drop_last": True}         
        self.train_loader, self.val_loader = make_data_loader(args, **self.kwargs)

        self.best = 0
        self.best_epoch = 0
        self.acc = []
        self.train_acc = []
        self.fp16 = False
        self.scaler = GradScaler(enabled=self.fp16)

        
    def train(self, epoch):
        running_loss = 0.0
        self.model.train()
        
        acc = 0
        tbar = tqdm(self.train_loader)
        m_dists = torch.tensor([])
        l = torch.tensor([])
        self.epoch = epoch
        total_sum = 0

        tbar.set_description("Training iMix, train_loss {}".format(""))
        
        #iMix + N-pairs
        for i, sample in enumerate(tbar):
            with autocast(enabled = self.fp16):
                img1, img2 = sample["image1"].cuda(), sample["image2"].cuda()
                bsz = img1.shape[0]
                labels = torch.arange(bsz).cuda()
                
                img1, _, _, lam, mix_index = mixup_data(img1, labels, 1.)
                
                z_i = self.model(img1)
                z_j = self.model(img2)
                
                logits = torch.div(torch.matmul(z_i, z_j.t()), 0.2) #Contrastive temp
                loss = lam * self.criterion(logits, labels) + (1 - lam) * self.criterion(logits, mix_index.cuda())
                loss = loss.mean()

                if i % 1 == 0:
                    tbar.set_description("Training iMix, train loss {:.2f}, lr {:.3f}".format(loss.item(), self.optimizer.param_groups[0]['lr']))
                    
            # compute gradient and do SGD step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad() 
        self.scheduler.step()
        print("Epoch: {0}".format(epoch))
        torch.save({'best':self.best, 'epoch':self.epoch, 'net':self.model.state_dict(), 'opt': self.optimizer.state_dict(), 'scheduler':self.scheduler.state_dict()}, os.path.join(self.args.save_dir, "last_model.pth.tar"))
    
def main():


    parser = argparse.ArgumentParser(description="iMix")
    parser.add_argument("--net", type=str, default="resnet18",
                        choices=["resnet18"],
                        help="net name")
    parser.add_argument("--dataset", type=str, default="irish", choices=["irish", "danish"])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument('--steps', type=int, default=[1000,1500], nargs='+', help='Epochs when to reduce lr')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.1, help="Multiplicative factor for lr decrease, default .1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="No cuda")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--proj-size", default=128, type=int)

    args = parser.parse_args()
    #For reproducibility purposes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        
    args.cuda = not args.no_cuda
    
    torch.manual_seed(args.seed)
    
    _trainer = Trainer(args)
    start_ep = 0 
    if args.resume is not None:
        l = torch.load(args.resume)
        start_ep = l['epoch']
        _trainer.best = l['best']
        _trainer.best_epoch = l['epoch']
        _trainer.model.load_state_dict(l['net'])
        _trainer.optimizer.load_state_dict(l['opt'])
        _trainer.scheduler.load_state_dict(l['scheduler'])
    for eps in range(start_ep, args.epochs):
        _trainer.train(eps)
        if not args.no_eval:
            _trainer.kNN()

if __name__ == "__main__":
   main()
