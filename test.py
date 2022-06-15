import os
import numpy as np
import tqdm
import pandas as pd
import cProfile
import faulthandler

import torch

from opts import parser
from model import ActionModifiers, Evaluator
from dataset import AdverbDataset

from sklearn.metrics import average_precision_score

def test(model, data_loader, test_set, evaluator):
    model.eval()

    y_true_adverb = np.zeros((len(test_set), len(test_set.adverbs)))
    y_score = np.zeros((len(test_set), len(test_set.adverbs)))
    y_score_antonym = np.zeros((len(test_set), len(test_set.adverbs)))

    test_data = []
    for idx, data in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        if args.gpu:
            data[0:4] = [d.cuda() for d in data[0:4]]
        clip_id = data[4]
        predictions, att = model(data)[1:3]
        adverb_gt, action_gt = data[1], data[2]
        _, action_gt_scores, antonym_action_gt_scores = evaluator.get_scores(predictions, action_gt, adverb_gt)
        for j in range(0, data[0].shape[0]):
            test_item = {}
            if att is not None:
                test_item['attention'] = att.mean(axis=1)[j].squeeze().cpu().numpy()
            test_item['clip_id'] = clip_id[j]
            test_item['adverb_gt'] = test_set.idx2adverb[adverb_gt[j].item()]
            test_item['action_gt'] = test_set.idx2action[action_gt[j].item()]

            y_true_adverb[idx*args.batch_size+j][adverb_gt[j].item()] = 1

            y_score[idx*args.batch_size+j] = np.array([action_gt_scores[j][test_set.pairs.index((adv, test_set.idx2action[action_gt[j].item()]))]
                                                       for adv in test_set.adverbs])
            y_score_antonym[idx*args.batch_size+j] = np.array([
                antonym_action_gt_scores[j][test_set.pairs.index((adv, test_set.idx2action[action_gt[j].item()]))]
                for adv in test_set.adverbs])
            test_item['adverb_predicted'] = test_set.idx2adverb[np.argmax(y_score_antonym[idx*args.batch_size+j])]
            test_item['adverb_order'] = [test_set.idx2adverb[adv_idx] for adv_idx in np.argsort(y_score[idx*args.batch_size+j])[::-1]]
            test_data.append(test_item)
    v2a_ant = (np.argmax(y_true_adverb, axis=1) == np.argmax(y_score_antonym, axis=1)).mean()
    per_adverb = {}
    for adv in test_set.adverb2idx.keys():
        inds = np.where(y_true_adverb[:,test_set.adverb2idx[adv]] == 1)
        per_ant = (np.argmax(y_true_adverb[inds], axis=1) == np.argmax(y_score_antonym[inds], axis=1)).mean()
        per_adverb[adv] = {'ant': per_ant}
    test_data_df = pd.DataFrame(test_data)
    test_data_df.to_pickle(os.path.join(os.path.dirname(args.load), 'predictions.pkl'))
    per_adverb_df = pd.DataFrame.from_dict(per_adverb, orient='index')
    per_adverb_df.to_csv(os.path.join(os.path.dirname(args.load), 'per_adverb.csv'))
    if not args.instance_av:
        print('Video-to-Adverb Antonym (av. per adverb): ', sum([per_adverb[adv]['ant'] for adv in per_adverb.keys()])/len(per_adverb.keys()))
    return v2a_ant

def main(args):
    test_set = AdverbDataset(args.data_dir, args.test_feature_dir, agg=args.temporal_agg, modality=args.modality,
                             window_size=args.t_test, adverb_filter=args.adverb_filter, phase='test', all_info=True, load_in_memory=False, unlabelled_ratio=args.unlabelled_ratio)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers)

    model = ActionModifiers(test_set, args)
    if args.gpu:
        model = model.cuda()

    evaluator = Evaluator(test_set, model)

    if args.load:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        print('loaded model from', os.path.basename(args.load))
    v2a_ant = test(model, test_loader, test_set, evaluator)
    if args.instance_av:
        print('Video-to-Adverb Antonym: %.3f'%v2a_ant)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.modality == 'both':
        args.modality = ['rgb', 'flow']
    else:
        args.modality = [args.modality]
    faulthandler.enable()
    main(args)
