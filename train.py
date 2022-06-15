import os
import shutil
import tqdm
import numpy as np
import torch
import torch.optim as optim
import time
import gc
from utils import save_args, introduce_adverbs, save_checkpoint, calculate_p1, calculate_mean_p1, AverageMeter

from opts import parser
from dataset import AdverbDataset
from model import ActionModifiers, Evaluator

from torch.utils.tensorboard import SummaryWriter


def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_args(args)

    train_set = AdverbDataset(args.data_dir, args.train_feature_dir, agg=args.temporal_agg,
                              modality=args.modality, window_size=args.t_train,
                              adverb_filter=args.adverb_filter, phase='train',
                              load_in_memory=args.load_in_memory,
                              unlabelled_ratio=args.unlabelled_ratio,
                              unlabelled_feature_dir=args.unlabelled_feature_dir)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.workers)
    test_set = AdverbDataset(args.data_dir, args.test_feature_dir, agg=args.temporal_agg,
                             modality=args.modality, window_size=args.t_test,
                             adverb_filter=args.adverb_filter, phase='test',
                             load_in_memory=args.load_in_memory,
                             unlabelled_ratio=args.unlabelled_ratio,
                             unlabelled_feature_dir=args.unlabelled_feature_dir)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    model = ActionModifiers(train_set, args).cuda()
    adverb_thresholds = None
    if args.adaptive_threshold:
        adverb_thresholds = torch.Tensor([args.pseudo_label_threshold]*len(model.dset.adverb2idx.keys())).cuda()

    evaluator = Evaluator(train_set, model)

    modifier_params = [param for name, param in model.named_parameters()
                       if ('action_modifiers' in name) and param.requires_grad]
    other_params = [param for name, param in model.named_parameters()
                    if ('action_modifiers' not in name) and param.requires_grad]
    if not args.pretrain_action:
        optim_params = [{'name': 'action_modifiers', 'params': modifier_params},
                        {'name': 'embedding', 'params': other_params}]
    else:
        optim_params = [{'name': 'action_modifiers', 'params': modifier_params, 'lr':0},
                        {'name': 'embedding', 'params': other_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    start_epoch = 0
    if args.load is not None:
        checkpoint = torch.load(args.load)
        pretrained_state_dict = checkpoint['net']
        model_state_dict = model.state_dict()
        pretrained_state_dict = {k:v for k, v in pretrained_state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)
        start_epoch = checkpoint['epoch']

    writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'log'))

    pseudo_start_epoch = args.pseudo_start_epoch
    pseudo_weight = args.pseudo_weight
    pseudo_always_action = args.pseudo_action_pretraining

    if args.pretrain_action:
        pseudo_weight = 0.0
    else:
        pseudo_weight = 0.0
    test(model, test_loader, evaluator, writer, start_epoch)
    for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
        if args.pretrain_action and epoch == args.adverb_start:
            introduce_adverbs(optimizer, args.lr)
        adverb_thresholds = train(model, train_loader, optimizer, writer, epoch, args.unlabelled_ratio, pseudo_weight, pseudo_always_action, args.num_pseudo_labelled, args.pseudo_selection, adverb_thresholds)
        if epoch % args.eval_interval == 0:
            test(model, test_loader, evaluator, writer, epoch)
        if epoch % args.save_interval == 0 and epoch > 0:
            save_checkpoint(model, epoch, args.checkpoint_dir)
        if epoch >= pseudo_start_epoch:
            pseudo_weight = args.pseudo_weight
    writer.close()

def pseudo_label_adverbs(model, features, actions, pad, num, method, threshold=0, adverb_thresholds=None):
    threshold_mask = torch.ones((num, actions.shape[0]), dtype=torch.bool)
    dummy_adverbs = torch.zeros(actions.shape).cuda()
    model.eval()
    data = [features, dummy_adverbs, actions, pad]
    predictions, attention = model(data)[1:3]
    combined_labels = np.array(list(predictions.keys()))
    predictions_tensor = torch.stack(list(predictions.values()))
    action_gt_mask = []
    actions_np = actions.cpu().numpy()
    for _adv, _act in model.dset.pairs:
        mask = model.dset.action2idx[_act]==actions_np
        action_gt_mask.append(torch.BoolTensor(mask))
    action_gt_mask = torch.stack(action_gt_mask, 0)
    predictions_tensor[~action_gt_mask] = -1e10 #Not masking max diff at start as gives 0
    if args.unseen_mask:
        unseen_mask = torch.zeros(predictions_tensor.shape[0], dtype=torch.bool)
        for i, (_adv, _act) in enumerate(model.dset.pairs):
            if (_act, _adv) in model.dset.unlabelled_pairs:
                unseen_mask[i] = True
        predictions_tensor[~unseen_mask, :] = -1e10
    reshaped_pred = predictions_tensor.reshape((int(len(model.dset.adverbs)/2), 2, len(model.dset.actions), -1))
    ant_pred = reshaped_pred[:, torch.LongTensor([1,0])]
    if method == 'closest':
        vals, predictions_ind = torch.topk(predictions_tensor, num, dim=0)
    elif method == 'diff':
        vals, predictions_ind = torch.topk((reshaped_pred-ant_pred).reshape(predictions_tensor.shape), num, dim=0)

    highest_pred = combined_labels[predictions_ind.cpu().numpy()]
    highest_pred = highest_pred[:,:,0]
    pseudo_adverbs = torch.tensor(np.vectorize(model.dset.adverb2idx.get)(highest_pred))

    conf = torch.zeros(2, 1, highest_pred.shape[-1])
    if threshold > 0:
        adv_vals = predictions_tensor.gather(0, predictions_ind)
        ant_vals = ant_pred.reshape(predictions_tensor.shape).gather(0,predictions_ind)
        scores = torch.stack([adv_vals, ant_vals])
        if args.conf_type == 'softmax':
            conf = torch.nn.functional.softmax(scores, 0)
        elif args.conf_type == 'margin':
            conf = scores - scores[torch.LongTensor([1,0])]
        if args.adaptive_threshold:
            adaptive_thres = adverb_thresholds[pseudo_adverbs]
            threshold_mask = conf[0] > adaptive_thres
        else:
            threshold_mask = conf[0] > threshold

    ants = np.vectorize(model.dset.antonyms.get)(highest_pred)
    unlabelled_neg_adverbs = torch.tensor(np.vectorize(model.dset.adverb2idx.get)(ants))
    model.train()
    return pseudo_adverbs, unlabelled_neg_adverbs, threshold_mask, attention, conf[0]


def train(model, train_loader, optimizer, writer, epoch, unlabelled_ratio, pseudo_loss_factor, pseudo_always_action, num_pseudo, pseudo_selection, adverb_thresholds):
    model.train()
    train_loss = 0.0
    act_loss = 0.0
    adv_loss = 0.0
    pseudo_train_loss = 0.0
    pseudo_act_loss = 0.0
    pseudo_adv_loss = 0.0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    start = time.time()
    all_pseudo_labels = torch.Tensor().long()
    unmasked_pseudo_labels = torch.Tensor().long()
    all_train_labels = torch.Tensor()
    all_actions = torch.Tensor().long().cuda()
    conf_sums = torch.Tensor([0]*len(model.dset.adverb2idx.keys())).cuda()
    for idx, data in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.unlabelled_ratio > 0:
            unlabelled_features = data[-4].cuda()
            unlabelled_actions = data[-3].cuda()
            unlabelled_pad = data[-2].cuda()
            unlabelled_neg_actions = data[-1].cuda()
            unlabelled_features = unlabelled_features.reshape(-1, *unlabelled_features.shape[2:])
            unlabelled_actions = unlabelled_actions.reshape(-1)
            unlabelled_neg_actions = unlabelled_neg_actions.reshape(-1)
            unlabelled_pad = unlabelled_pad.reshape(-1)
            data = [d.cuda() for d in data[:-4]]
        else:
            data = [d.cuda() for d in data]
        data_time.update(time.time() - start)
        all_loss = model(data)[0]
        loss = sum(all_loss)
        pseudo_loss = 0
        iter_pseudo_act_loss = 0
        if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action):
            pseudo_labelled_adverbs, unlabelled_neg_adverbs, threshold_mask, attention, adverb_scores = pseudo_label_adverbs(model, unlabelled_features, unlabelled_actions, unlabelled_pad, num_pseudo, pseudo_selection, args.pseudo_label_threshold, adverb_thresholds=adverb_thresholds)
            if args.adaptive_threshold:
                for j in range(adverb_scores.shape[1]):
                    conf_sums[pseudo_labelled_adverbs[:,j]] += adverb_scores[:,j]
            unmasked_pseudo_labels = torch.cat([unmasked_pseudo_labels, pseudo_labelled_adverbs[threshold_mask].reshape(-1)])
            all_pseudo_labels = torch.cat([all_pseudo_labels, pseudo_labelled_adverbs.reshape(-1)])
            all_actions = torch.cat([all_actions, unlabelled_actions.repeat_interleave(num_pseudo)])
            for i in range(0, pseudo_labelled_adverbs.shape[0]):
                unlabelled_data = [unlabelled_features, pseudo_labelled_adverbs[i].cuda(), unlabelled_actions, unlabelled_pad, unlabelled_neg_adverbs[i].cuda(), unlabelled_neg_actions]
                if args.pseudo_label_threshold > 0:
                    all_pseudo_loss = model(unlabelled_data, threshold_adverbs=threshold_mask[i])[0]
                else:
                    all_pseudo_loss = model(unlabelled_data)[0]
                pseudo_loss += sum(all_pseudo_loss)
                iter_pseudo_act_loss += all_pseudo_loss[0].item()
                pseudo_adv_loss += all_pseudo_loss[1].item()
            pseudo_train_loss += pseudo_loss.item()
            pseudo_act_loss += iter_pseudo_act_loss
            if pseudo_loss_factor > 0:
                total_loss = loss + (pseudo_loss/pseudo_labelled_adverbs.shape[0]) * pseudo_loss_factor
            else:
                total_loss = loss + (iter_pseudo_act_loss/pseudo_labelled_adverbs.shape[0])
        else:
            total_loss = loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += loss.item()
        act_loss += all_loss[0].item()
        adv_loss += all_loss[1].item() ##should be introduced after action mods have started training

        batch_time.update(time.time() - start - data_time.val)
        start = time.time()
        if epoch == 0:
            all_train_labels = torch.cat([all_train_labels, data[1].cpu()])

    ## Updated adverb thresholds
    if args.adaptive_threshold and unlabelled_ratio > 0 and pseudo_loss_factor > 0:
        class_counts = torch.bincount(all_pseudo_labels)
        av_count = class_counts.sum()/class_counts.shape[0]
        adverb_thresholds = ((conf_sums/av_count)**args.smoothing)*args.pseudo_label_threshold

    train_loss /= len(train_loader)
    act_loss /= len(train_loader)
    adv_loss /= len(train_loader)
    if args.unlabelled_ratio > 0 and (pseudo_loss_factor > 0 or pseudo_always_action):
        pseudo_train_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        pseudo_act_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        pseudo_adv_loss /= (len(train_loader)*pseudo_labelled_adverbs.shape[0])
        if unmasked_pseudo_labels.shape[0] > 0:
            writer.add_histogram('PseudoLabelDist', unmasked_pseudo_labels, epoch)
        writer.add_scalar('PseudoLabel/AboveThreshold', threshold_mask.sum(), epoch)
        if args.adaptive_threshold:
            for i in range(adverb_thresholds.shape[0]):
                writer.add_scalar('PseduoLabelThresholds/' + str(i), adverb_thresholds[i], epoch)
    writer.add_scalar('Loss/Train/Total', train_loss, epoch)
    writer.add_scalar('Loss/Train/Action', act_loss, epoch)
    writer.add_scalar('Loss/Train/Adverb', adv_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoTotal', pseudo_train_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoAction', pseudo_act_loss, epoch)
    writer.add_scalar('Loss/Train/PseudoAdverb', pseudo_adv_loss, epoch)

    if epoch == 0:
        writer.add_histogram('TrainLabelDist', all_train_labels, epoch)
    print('E: %d | L: %.2E | L_act: %.2E | L_adv: %.2E | Batch Time: %.2E | Data Time: %.2E'%(epoch, train_loss, act_loss, adv_loss, batch_time.avg, data_time.avg))
    gc.collect()
    return adverb_thresholds

def test(model, test_loader, evaluator, writer, epoch):
    model.eval()
    accuracies = []
    all_antonym_action_gt_scores = torch.Tensor()
    all_adverb_gt = torch.Tensor().cuda()
    for idx, data in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        data = [d.cuda() for d in data]
        predictions = model(data)[1]
        adverb_gt, action_gt = data[1], data[2]
        scores, action_gt_scores, antonym_action_gt_scores = evaluator.get_scores(predictions, action_gt, adverb_gt)
        all_antonym_action_gt_scores = torch.cat([all_antonym_action_gt_scores, antonym_action_gt_scores])
        all_adverb_gt = torch.cat([all_adverb_gt, adverb_gt])
        acc = calculate_p1(model.dset, antonym_action_gt_scores, adverb_gt)
        print('E %d | Video-to-Adverb Antonym P@1: %.3f'%(epoch, acc))
        accuracies.append(acc)
    acc_mean = calculate_mean_p1(model.dset, all_antonym_action_gt_scores, all_adverb_gt)
    writer.add_scalar('Acc/Test/Video-to-Adverb Antonym', sum(accuracies)/len(accuracies), epoch)
    writer.add_scalar('Acc/Test/Video-to-Adverb Antonym Mean', acc_mean, epoch)

def calculate_p1_action(dset, scores, action_gt):
    pair_pred = np.argmax(scores.numpy(), axis=1)
    action_pred = [dset.action2idx[dset.pairs[pred][1]] for pred in pair_pred]
    acc = (action_pred == action_gt.cpu().numpy()).mean()
    return acc

if __name__ == '__main__':
    args = parser.parse_args()
    if args.modality == 'both':
        args.modality = ['rgb', 'flow']
    else:
        args.modality = [args.modality]
    main(args)
