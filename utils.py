import os
import shutil
import torch
import numpy as np

def save_args(args):
    shutil.copy('train.py', args.checkpoint_dir)
    #shutil.copy('models.py', args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

def introduce_adverbs(optimizer, lr):
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'action_modifiers':
            param_group['lr'] = lr * 0.1 
        else:
            param_group['lr'] = lr * 0.1

def save_checkpoint(model, epoch, checkpoint_dir):
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(checkpoint_dir, 'ckpt_E_%d'%(epoch)))

def calculate_p1(dset, scores, adverb_gt):
    pair_pred = np.argmax(scores.numpy(), axis=1)
    adverb_pred = [dset.adverb2idx[dset.pairs[pred][0]] for pred in pair_pred]
    acc = (adverb_pred == adverb_gt.cpu().numpy()).mean() ##need way to get pair gt or convert from pair gt to adverb gt
    return acc

def calculate_mean_p1(dset, scores, adverb_gt):
    pair_pred = np.argmax(scores.numpy(), axis=1)
    adverb_pred = [dset.adverb2idx[dset.pairs[pred][0]] for pred in pair_pred]
    accs = (adverb_pred == adverb_gt.cpu().numpy())
    adverb_gt_cpu = adverb_gt.cpu().numpy()
    per_class = [[accs[i] for i in range(scores.shape[0]) if adverb_gt_cpu[i] == adv] for adv in dset.adverb2idx.values()]
    per_class_accs = [sum(l)/float(len(l)) for l in per_class if len(l) > 0]
    acc = sum(per_class_accs)/len(per_class_accs)
    return acc


class AverageMeter(object):
    def __init__(self):
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
