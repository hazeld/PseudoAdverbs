import os
import math
import itertools
import random
import numpy as np
import pandas as pd
import tqdm
import torch.utils.data as data

class AdverbDataset(data.Dataset):

    def __init__(self, data_dir, feature_dir, agg='sdp', modality=['rgb', 'flow'], window_size=None,
                 adverb_filter=None, phase='test', action_key='clustered_action',
                 adverb_key='clustered_adverb', all_info=False,
                 load_in_memory=True, unlabelled_ratio=0,
                 train_file='train.csv',
                 test_file='test.csv', unlabelled_file='unlabelled.csv',
                 unlabelled_feature_dir=None):
        self.train_file = train_file
        self.test_file = test_file
        self.unlabelled_file = unlabelled_file
        if unlabelled_feature_dir is None:
            self.unlabelled_feature_dir = feature_dir
        else:
            self.unlabelled_feature_dir = unlabelled_feature_dir
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.agg = agg
        self.modality = modality
        self.window_size = window_size
        self.phase = phase
        self.action_key = action_key
        self.adverb_key = adverb_key
        self.all_info = all_info
        self.load_in_memory = load_in_memory
        self.unlabelled_ratio = unlabelled_ratio

        if self.unlabelled_ratio > 0:
            self.adverbs, self.actions, self.train_list, self.test_list, self.unlabelled_list = self._parse_list(adverb_filter)
        else:
            self.adverbs, self.actions, self.train_list, self.test_list = self._parse_list(adverb_filter)

        self.adverbs, self.antonyms = self._add_antonyms(self.adverbs) ## antonyms necessary for training

        self.pairs = list(itertools.product(self.adverbs, self.actions))
        if self.unlabelled_ratio > 0:
            self.unlabelled_pairs = list(set(list(self.unlabelled_list[['clustered_action', 'clustered_adverb']].itertuples(index=False, name=None))))
            for pair in self.unlabelled_pairs:
                if not isinstance(pair[1], str):
                    continue
                if (pair[0], self.antonyms[pair[1]]) not in self.unlabelled_pairs:
                    self.unlabelled_pairs.append((pair[0], self.antonyms[pair[1]]))

        assert pd.merge(self.train_list, self.test_list, how='inner', on=['action', 'adverb', 'clip_id']).shape[0] == 0, 'train and test are not mutually exclusive ' + str(pd.merge(self.train_list, self.test_list, how='inner', on=['action', 'adverb', 'clip_id']))

        self.data = self.train_list if self.phase == 'train' else self.test_list
        self.adverb2idx = {adverb: idx for idx, adverb in enumerate(self.adverbs)}
        self.idx2adverb = {v:k for k, v in self.adverb2idx.items()}
        self.action2idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx2action = {v:k for k, v in self.action2idx.items()}
        if 'start_time' in self.data.columns:
            self.max_temporal_dim = int(max(self.data['end_time'] - self.data['start_time']))
        else:
            self.max_temporal_dim = window_size
        if self.load_in_memory:
            self.feature_list = self._load_all_features(self.data, self.feature_dir)
            self.feature_dim = self.feature_list[0][0].shape[-1]
            print('%d features loaded'%(len(self.feature_list)))
            if self.unlabelled_ratio > 0:
                self.unlabelled_feature_list = self._load_all_features(self.unlabelled_list, self.unlabelled_feature_dir)
        else:
            first_item = self._load_single_feature(0, self.data.iloc[0], self.feature_dir)
            self.feature_dim = first_item[0].shape[-1]

    def feature_pad(self, features, pad_length):
        current_temp_dim = features.shape[0]
        if current_temp_dim == pad_length:
            return features
        feature_dim = features.shape[1]
        padded_features = np.zeros((pad_length, feature_dim), dtype=np.float32)
        padded_features[:current_temp_dim] = features
        return padded_features

    def _get_feature_filename(self, x, modality):
        return '_'.join((x['clip_id'], modality + '.npz'))

    def _get_window(self, features):
        if self.window_size:
            features = [feature[math.ceil(feature.shape[0]/2-self.window_size/2):
                                math.ceil(feature.shape[0]/2+self.window_size/2)]
                        for feature in features]
        return features

    def _load_feature_from_file(self, i, x, feature_dir):
        features = [np.load(os.path.join(feature_dir, self._get_feature_filename(x, modality)))['arr_0'] for modality in self.modality]
        features = self._get_window(features)
        data_tuple = (features, x[self.adverb_key], x[self.action_key], x['clip_id'])
        return data_tuple

    def _load_single_feature(self, i, x, feature_dir):
        data_tuple = self._load_feature_from_file(i, x, feature_dir)
        feature_dim = data_tuple[0][0].shape
        features = data_tuple[0]
        adv = data_tuple[1]
        act = data_tuple[2]
        clip_id = data_tuple[3]
        ## deal with unequal lengths
        feature_dims = [feature.shape for feature in data_tuple[0]]
        min_feature_dim = min([feature_dim[0] for feature_dim in feature_dims])
        max_feature_dim = max([feature_dim[0] for feature_dim in feature_dims])
        ##deal with different temporal length
        features = [feature[:min_feature_dim] for feature in features]

        if len(feature_dim) > 2:
            features = [feature[:,0].reshape((-1, feature_dim[-1]))
                              for feature in features]
        if self.agg == 'single':
            data_tuple = (np.concatenate([feature[math.ceil(feature.shape[0]/2)] for feature in features]),
                                  adv, act, clip_id)
        elif self.agg == 'average':
            data_tuple = (np.concatenate([feature.mean(axis=0) for feature in features]), adv, act, clip_id)
        elif self.agg == 'sdp':
            max_dim = self.max_temporal_dim
            if len(feature_dim) > 2:
                max_dim = max_dim * feature_dim[-3] * feature_dim[-2]
            data_tuple = (self.feature_pad(np.concatenate([feature for feature in features], axis=1),
                                              max_dim),
                                  adv, act, clip_id, max_dim-features[0].shape[0])
        else:
            print("Error: temporal aggregation method not supported")
            exit(0)
        return data_tuple


    def _load_all_features(self, data, feature_dir):
        print("Loading features")
        feature_list = []
        for i, x in tqdm.tqdm(data.iterrows(), total=len(data)):
            data_tuple = self._load_feature_from_file(i, x, feature_dir)
            feature_list.append(data_tuple)
        feature_dims = [[feature.shape for feature in vid_data[0]] for vid_data in feature_list]
        min_feature_dims = [min([feature_dim[0] for feature_dim in vid_feature_dims]) for vid_feature_dims in feature_dims]
        max_feature_dims = [max([feature_dim[0] for feature_dim in vid_feature_dims]) for vid_feature_dims in feature_dims]
        unequal = [(feature_list[i][3],  max_feature_dims[i]-min_feature_dims[i]) for i in range(len(feature_list)) if max_feature_dims[i]-min_feature_dims[i] != 0]
        ##deal with different temporal length
        feature_list = [([feature[:min_feature_dims[i]]
                              for feature in features], adv, act, clip_id)
                            for i, (features, adv, act, clip_id) in enumerate(feature_list)]

        if len(feature_dims[0][0]) > 2:
            feature_list = [([feature.mean(axis=1).reshape((-1, feature_dims[0][0][-1]))
                              for feature in features], adv, act, clip_id)
                            for (features, adv, act, clip_id) in feature_list]
        print("Aggregating")
        if self.agg == 'single':
            feature_list = [(np.concatenate([feature[math.ceil(feature.shape[0]/2)] for feature in features]),
                                  adv, act, clip_id)
                                 for (features, adv, act, clip_id) in feature_list]
        elif self.agg == 'average':
            feature_list = [(np.concatenate([feature.mean(axis=0) for feature in features]), adv, act, clip_id)
                                 for (features, adv, act, clip_id) in feature_list]
        elif self.agg == 'sdp':
            max_temporal_dim = max([feature_list[i][0][0].shape[0] for i in range(len(feature_list))])
            feature_list = [(self.feature_pad(np.concatenate([feature for feature in features], axis=1),
                                              max_temporal_dim),
                                  adv, act, clip_id, max_temporal_dim-features[0].shape[0]) for (features, adv, act, clip_id,) in feature_list]
            print(feature_list[0][0].shape)
        else:
            print("Error: temporal aggregation method not supported")
            exit(0)
        return feature_list


    def _add_antonyms(self, adverb_list):
        antonyms_df = pd.read_csv(os.path.join(self.data_dir, 'antonyms.csv'))
        adverbs = []
        antonyms = {}
        for i, row in antonyms_df.iterrows():
            if row['adverb'] in adverb_list:
                if row['adverb'] not in adverbs:
                    adverbs.append(row['adverb'])
                if row['antonym'] not in adverbs:
                    adverbs.append(row['antonym'])
            antonyms[row['adverb']] = row['antonym']
        return adverbs, antonyms

    def _parse_list(self, adverb_filter):
        def parse_pairs(filename):
            pairs_df = pd.read_csv(filename)
            if adverb_filter is not None:
                pairs_df = pairs_df[pairs_df[self.adverb_key].isin(adverb_filter)]
            mods = pairs_df[self.adverb_key].unique().tolist()
            acts = pairs_df[self.action_key].unique().tolist()
            return mods, acts, pairs_df

        train_mods, train_acts, train_list = parse_pairs(os.path.join(self.data_dir, self.train_file))
        test_mods, test_acts, test_list = parse_pairs(os.path.join(self.data_dir, self.test_file))
        if self.unlabelled_ratio > 0:
            _, unlabelled_acts, unlabelled_list = parse_pairs(os.path.join(self.data_dir, self.unlabelled_file))
        all_mods = sorted(list(set(train_mods+test_mods)))
        if self.unlabelled_ratio > 0:
            all_acts = sorted(list(set(train_acts+test_acts+unlabelled_acts)))
        else:
            all_acts = sorted(list(set(train_acts+test_acts)))
        unique_pairs = train_list.drop_duplicates(subset=['clustered_action', 'clustered_adverb'])
        self.train_pairs = list(zip(list(unique_pairs['clustered_adverb']), list(unique_pairs['clustered_action'])))

        if self.unlabelled_ratio > 0:
            return all_mods, all_acts, train_list, test_list, unlabelled_list
        else:
            return all_mods, all_acts, train_list, test_list

    def sample_negative_action_weighted(self, adverb, action):
        _, new_action = self.train_pairs[np.random.choice(len(self.train_pairs))]
        if new_action==action:
            return self.sample_negative_action_weighted(adverb, action)
        return new_action


    def sample_negative_action(self, action):
        new_action = self.actions[np.random.choice(len(self.actions))]
        if new_action==action:
            return self.sample_negative_action(action)
        return new_action

    def sample_adverb(self, adverb):
        new_adverb = self.adverbs[np.random.choice(len(self.adverbs))]
        return new_adverb

    def __getitem__(self, index):
        if self.load_in_memory:
            item_feature = self.feature_list[index]
        else:
            ind_data = self.data.iloc[index]
            item_feature = self._load_single_feature(index, ind_data, self.feature_dir)
        if self.unlabelled_ratio > 0:
            unlabelled_inds = [random.randint(0, len(self.unlabelled_list)-1) for i in range(0, self.unlabelled_ratio)]
            if self.load_in_memory:
                unlabelled_features = [self.unlabelled_feature_list[u_ind] for u_ind in unlabelled_inds]
            else:
                inds_data = [self.unlabelled_list.iloc[u_ind] for u_ind in unlabelled_inds]
                unlabelled_features = [self._load_single_feature(unlabelled_inds[i], inds_data[i], self.unlabelled_feature_dir) for i in range(0, self.unlabelled_ratio)]
        feature, adverb, action = item_feature[0:3]
        data = [feature, self.adverb2idx[adverb], self.action2idx[action]]
        if self.agg == 'sdp':
            pad = item_feature[4]
            data += [pad]
        else:
            data += [0]
        if self.phase == 'train':
            neg_adverb = self.adverb2idx[self.antonyms[adverb]]
            neg_action = self.action2idx[self.sample_negative_action(action)]
            data += [neg_adverb, neg_action]
            if self.unlabelled_ratio > 0:
                u_feats = np.array([u_feature[0] for u_feature in unlabelled_features])
                u_acts = np.array([self.action2idx[u_feature[2]] for u_feature in unlabelled_features])
                u_pad = np.array([u_feature[4] for u_feature in unlabelled_features])
                u_neg_acts = np.array([self.action2idx[self.sample_negative_action(u_feature[2])] for u_feature in unlabelled_features])
                data += [u_feats, u_acts, u_pad, u_neg_acts]
        if self.all_info:
            clip_id = item_feature[3]
            data += [clip_id]
        return data

    def __len__(self):
        return len(self.data)
