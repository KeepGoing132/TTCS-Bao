"""
Utility functions for training and evaluation
Includes: metrics, logging, checkpointing
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import config


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Displays progress of training"""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class RecorderMeter:
    """Records and plots training metrics"""
    
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.reset()
    
    def reset(self):
        self.values = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update metrics for an epoch"""
        self.values['train_loss'].append(train_loss)
        self.values['train_acc'].append(train_acc)
        self.values['val_loss'].append(val_loss)
        self.values['val_acc'].append(val_acc)
    
    def plot_curve(self, save_path):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.values['train_loss']) + 1)
        
        # Loss curve
        ax1.plot(epochs, self.values['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.values['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, self.values['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.values['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100)
        plt.close()
    
    def to(self):
        """Convert to tensor if needed"""
        return self


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, args, save_dir=None):
    """Save checkpoint"""
    if save_dir is None:
        save_dir = config.CHECKPOINT_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    
    if hasattr(args, 'checkpoint_path'):
        filepath = args.checkpoint_path
    else:
        now = datetime.now()
        time_str = now.strftime("[%m-%d]-[%H-%M]-")
        filepath = os.path.join(save_dir, f"{time_str}model.pth.tar")
    
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = filepath.replace('model.pth.tar', 'model_best.pth.tar')
        torch.save(state, best_filepath)
        print(f"Saved best checkpoint to {best_filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=None):
    """Load checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return model, optimizer, start_epoch, best_acc


def adjust_learning_rate(optimizer, epoch, args):
    """Adjust learning rate based on schedule"""
    if hasattr(args, 'lr_factor') and hasattr(args, 'lr_patience'):
        lr = args.lr * (args.lr_factor ** (epoch // args.lr_patience))
    else:
        # Use config values
        lr = config.LR * (config.LR_FACTOR ** (epoch // config.LR_PATIENCE))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def create_logger(log_dir=None):
    """Create logger for training"""
    if log_dir is None:
        log_dir = config.LOG_DIR
    
    os.makedirs(log_dir, exist_ok=True)
    
    now = datetime.now()
    time_str = now.strftime("[%m-%d]-[%H-%M]-")
    log_path = os.path.join(log_dir, f"{time_str}log.txt")
    
    return log_path


def log_message(log_path, message):
    """Append message to log file"""
    with open(log_path, 'a') as f:
        f.write(message + '\n')
    print(message)
