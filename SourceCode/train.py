
import os
import argparse
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import config
from dataset import get_dataloaders
from models import get_model
from losses import get_criterion
from utils import (
    AverageMeter, ProgressMeter, RecorderMeter,
    accuracy, save_checkpoint, load_checkpoint,
    adjust_learning_rate, create_logger, log_message
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train emotion recognition model on RAF-DB with ResNet50'
    )
    parser.add_argument('--batch-size', default=config.BATCH_SIZE, type=int,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', default=config.EPOCHS, type=int,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', default=config.LR, type=float,
                        help='Initial learning rate (default: 0.01)')
    parser.add_argument('--workers', default=config.NUM_WORKERS, type=int,
                        help='Number of workers (default: 2)')
    parser.add_argument('--resume', default=None, type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU device id (default: 0)')
    
    return parser.parse_args()


def train_epoch(train_loader, model, criterion, optimizer, device, epoch, args, log_path):
    model.train()
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1],
        prefix=f'Epoch: [{epoch+1}]'
    )
    
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        
        output = model(images)
        loss = criterion(output, target)
        
        if config.USE_REGULARIZATION:
            reg_factor = config.REGULARIZATION_FACTOR
            if config.REGULARIZATION_TYPE == 'l2':
                l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())
                loss = loss + reg_factor * l2_reg
            elif config.REGULARIZATION_TYPE == 'l1':
                l1_reg = sum(torch.sum(torch.abs(p)) for p in model.parameters())
                loss = loss + reg_factor * l1_reg
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        
        if (i + 1) % config.PRINT_FREQ == 0:
            progress.display(i + 1)
            lr = optimizer.param_groups[0]['lr']
            print(f"    ├─ Accuracy: {top1.avg:.2f}% | LR: {lr:.2e} | "
                  f"Reg: {config.REGULARIZATION_TYPE} | Dropout: {config.USE_DROPOUT} ({config.DROPOUT_PROB})")
    
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, device, epoch, log_path=None):
    model.eval()
    
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.2f')
    
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(device)
            target = target.to(device)
            
            output = model(images)
            loss = criterion(output, target)
            
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
    
    return losses.avg, top1.avg


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    log_path = create_logger()
    now = datetime.datetime.now()
    log_message(log_path, f"Training started: {now.strftime('%m-%d %H:%M')}")
    log_message(log_path, f"Model: ResNet50")
    log_message(log_path, f"Batch size: {args.batch_size}")
    log_message(log_path, f"Epochs: {args.epochs}")
    log_message(log_path, f"Learning rate: {args.lr}")
    
    best_acc = 0
    start_epoch = 0
    
    print("Creating ResNet50 model...")
    model = get_model(
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    model = model.to(device)
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    
    criterion = get_criterion()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            model, optimizer, start_epoch, best_acc = load_checkpoint(
                args.resume, model, optimizer, device
            )
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    recorder = RecorderMeter(args.epochs)
    
    if args.evaluate:
        print("Evaluating model...")
        val_acc, val_loss = validate(val_loader, model, criterion, device, 0, log_path)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        return
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        log_message(log_path, f"Epoch [{epoch+1}/{args.epochs}] LR: {current_lr:.6f}")
        
        train_loss, train_acc = train_epoch(
            train_loader, model, criterion, optimizer, device, epoch, args, log_path
        )
        
        val_loss, val_acc = validate(
            val_loader, model, criterion, device, epoch, log_path
        )
        
        recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'recorder': recorder
        }
        
        if (epoch + 1) % config.SAVE_FREQ == 0 or is_best:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            save_checkpoint(checkpoint, is_best, args)
        
        curve_path = os.path.join(config.LOG_DIR, 'training_curve.png')
        recorder.plot_curve(curve_path)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        log_message(log_path, f"Epoch time: {epoch_time:.2f}s, Best Acc: {best_acc:.2f}%\n")
        
        print(f"\n{'='*100}")
        print(f"Epoch [{epoch+1}/{args.epochs}] | Time: {epoch_time:.2f}s")
        print(f"  📊 Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  📊 Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}% | Best: {best_acc:.2f}%")
        print(f"  ⚙️  LR={args.lr:.2e}, BatchSize={args.batch_size}, Epochs={args.epochs}")
        print(f"  🔧 Regularization: {config.REGULARIZATION_TYPE} (factor={config.REGULARIZATION_FACTOR})")
        print(f"  💧 Dropout: {config.USE_DROPOUT} (rate={config.DROPOUT_PROB})")
        print(f"  🖥️  Device: {device}")
        print(f"{'='*100}\n")
    
    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    log_message(log_path, f"Training completed! Best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
