import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/',
                    help='directory containing the training test splits and list of antonyms')
parser.add_argument('--train-feature-dir', default='', help='directory containing the rgb and flow features')
parser.add_argument('--test-feature-dir', default='features/', help='directory containing the rgb and flow features for the test set')
parser.add_argument('--checkpoint-dir', default='tmp/', help='directory to save checkpoints and tensorflow logs to')
parser.add_argument('--load', default=None, help='path to checkpoint to load')
parser.add_argument('--adverb-filter', nargs='+', default=None, help='select adverbs to train')

## model parameters
parser.add_argument('--emb-dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--no-glove-init', dest='glove_init', action='store_false',
                    help='don\'t initialize the action embeddings with word vectors')
parser.add_argument('--temporal-agg', default='sdp', choices=['single', 'average', 'sdp'],
                    help='method to aggregate the features in the window of size T')
parser.add_argument('--modality', default='both', choices=['rgb', 'flow', 'both', 'audio', 'all', 'rgb_audio'],
                    help='modalities used to train the model')
parser.add_argument('--t_train', type=int, default=None, help='size of the temporal window used around the weak timestamp')
parser.add_argument('--t_test', type=int, default=None, help='size of the temporal window used around the weak timestamp')
parser.add_argument('--no-pretrain-action', dest='pretrain_action', action='store_false',
                    help='don\'t first train the actions without action modifiers')
parser.add_argument('--adverb-start', type=int, default=200,
                    help='epoch to introduce action modifiers at if action if \'pretrain action\' is True')

## optimization
parser.add_argument('--batch-size', type=int, default=128, help='number of data points in a batch')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--workers', type=int, default=8, help='number of workers used to load data')
parser.add_argument('--save-interval', type=int, default=100, help='number of epochs to save a checkpoint after')
parser.add_argument('--eval-interval', type=int, default=20, help='number of epochs to test after')
parser.add_argument('--max-epochs', type=int, default=800, help='max_epochs to run the model for')
parser.add_argument('--no-gpu', dest='gpu', action='store_false', help='run only on the cpu')
parser.add_argument('--no-load-in-memory', dest='load_in_memory', action='store_false', help='load features every interation')
#pseudo adverb flags
parser.add_argument('--unlabelled-ratio', type=int, default=0)
parser.add_argument('--dummy-adverb', action='store_true', default=False)
parser.add_argument('--pseudo-action-pretraining', action='store_true', default=False)
parser.add_argument('--num-pseudo-labelled', type=int, default=5, help='k in the paper')
parser.add_argument('--pseudo-selection', type=str, choices=['closest', 'diff'], default='diff')
parser.add_argument('--pseudo-start-epoch', type=int, default=200)
parser.add_argument('--pseudo-weight', type=float, default=1.0)
parser.add_argument('--pseudo-label-threshold', type=float, default=0, help='tau in the paper')
parser.add_argument('--unseen-mask', default=False, action='store_true')
parser.add_argument('--unlabelled-feature-dir', default=None)
parser.add_argument('--adaptive-threshold', default=False, action='store_true')
parser.add_argument('--smoothing', type=float, default=0.05, help='lambda in the paper')
parser.add_argument('--conf-type', default='softmax', choices=['softmax', 'margin'])

parser.add_argument('--instance-av', default=False, action='store_true', help='To show average over instances in testing rather than per adverb class')
