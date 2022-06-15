# PseudoAdverbs

Datasets & Code for the CVPR 2022 paper 'How Do You Do It? Fine-Grained Action Understanding with Pseudo-Adverbs'

## Datasets

The three datasets proposed in this work can be found in the `datasets` folder. Each dataset has its own folder with an `annotations.csv` file and an `adverbs.csv` file. 

`adverbs.csv` contains the list of adverbs annotated in the dataset and their corresponding antonyms.

`annotations.csv` contains the YouTube video ID, the start and end time of the video clip and the action-adverb annotation. It has the following columns:

| Column Name   | Type          | Example | Description |
| ------------- |:-------------:| -------:| -----------:|
| clip_id       | string | video663 | Unique id for the video clip corresponding to the original dataset |
| youtube_id    | string | S7wF6S5ywo4 | YouTube id for the full video |
| start_time    | float | 3.22 | Value in seconds of the start time for the video clip |
| end_time      | float |  19.07 | Value in seconds of the end time for the video clip |
| action        | string | drive | The original action from the caption |
| adverb        | string | rapidly | The original adverb from the caption |
| caption       | string | a blue lamborghini drives rapidly down a busy street | The full caption for the video clip |
| clustered_adverb | string | drive | Annotated adverb |
| clustered_action | string | quickly | Annotated action |

Videos can be downloaded with `python utils/download_videos.py datasets/<dataset_name>/annotations.csv <download_dir>`

Extracted features can be downloaded here: https://drive.google.com/drive/folders/1gdF_Hp7hBmlRHqs9TG0pehZzSzaZw5Bc?usp=sharing

## Code

Trained models can be downloaded here: https://drive.google.com/file/d/1BqrzjdFQ5bqrhlWakiNQdi_5NE7h64Nb/view?usp=sharing

For models trained on VATEX we provide models trained for action embedding for the first 200 epochs here: https://drive.google.com/file/d/13aTARYglJfHa11SVRO9BSC6V3eEyIBdS/view?usp=sharing

Below we give the commands needed to train and test models for the different tasks and datasets.

### Seen Compositions

#### VATEX Adverbs

<strong>Train</strong>
```
python train.py --data-dir splits/with_unlabelled/seen_compositions/vatex_adverbs_5/ --train-feature-dir data/VATEX_Adverbs/features/ --test-feature-dir data/VATEX_Adverbs/features/ --checkpoint-dir checkpoints/vatex_adverbs_5_pseudo_adverbs/ --modality both --unlabelled-ratio 19 --load checkpoints/action_pretraining/vatex_adverbs_5_action_pretraining --pseudo-label-threshold 0.6 --smoothing 0.1 --adaptive-threshold
```

<strong>Test</strong>
```
python test.py --unlabelled-ratio 1 --data-dir splits/with_unlabelled/seen_compositions/vatex_adverbs_5/ --test-feature-dir data/VATEX_Adverbs/features/ --load checkpoints/vatex_adverbs_5_pseudo_adverbs/ckpt_E_1000
```

#### HowTo100M Adverbs

<strong>Train</strong>
```
python train.py --data-dir splits/with_unlabelled/seen_compositions/howto100m_adverbs_10/ --train-feature-dir data/HowTo100M_Adverbs/features/ --test-feature-dir data/HowTo100M_Adverbs/features/ --checkpoint-dir checkpoints/howto100m_adverbs_10_pseudo_adverbs/ --modality both --unlabelled-ratio 9 --pseduo-label-threshold 0.6 --smoothing 0.1 --adaptive-threshold --t_train 20 --t_test 20 --num-pseudo-labelled 3 --pseudo-action-pretraining --max-epochs 1000
```

<strong>Test</strong>
```
python test.py --unlabelled-ratio 1 --t_test 20 --data-dir splits/with_unlabelled/seen_compositions/howto100m_adverbs_10/ --test-feature-dir data/HowTo100M_Adverbs/features/ --load checkpoints/howto100m_adverbs_10_pseudo_adverbs/ckpt_E_1000 --instance-av
```

### Unseen Compositions

<strong>Train</strong>
```
python train.py --data-dir splits/with_unlabelled/unseen_compositions/vatex/ --train-feature-dir data/VATEX_Adverbs/features/ --test-feature-dir data/VATEX_Adverbs/features/ --checkpoint-dir checkpoints/vatex_adverbs_unseen_pseudo_adverbs/ --modality both --unlabelled-ratio 1 --load checkpoints/action_pretraining/vatex_adverbs_unseen_action_pretraining --pseudo-label-threshold 0.6 --smoothing 0.1 --adaptive-threshold
```

<strong>Test</strong>
```
python test.py --unlabelled-ratio 1 --data-dir splits/with_unlabelled/unseen_compositions/vatex/ --test-feature-dir data/VATEX_Adverbs/features/ --load checkpoints/vatex_adverbs_unseen_pseudo_adverbs/ckpt_E_1000
```

### Unseen Domains

<strong>Train</strong>
```
python train.py --data-dir splits/with_unlabelled/unseen_domains/vatex2msrvtt/ --train-feature-dir data/VATEX_Adverbs/features/ --unlabelled-feature-dir data/MSR-VTT_Adverbs/features/ --test-feature-dir data/MSR-VTT_Adverbs/features/ --checkpoint-dir checkpoints/vatex2msrvtt_pseudo_adverbs/ --modality both --unlabelled-ratio 1 --load checkpoints/action_pretraining/vatex2msrvtt_pseudo_adverbs_action_pretraining --pseudo-label-threshold 0.6 --smoothing 0.1 --adaptive-threshold
```

<strong>Test</strong>
```
python test.py --unlabelled-ratio 1 --data-dir splits/with_unlabelled/unseen_domains/vatex2msrvtt/ --test-feature-dir data/MSR-VTT_Adverbs/features/ --load checkpoints/vatex2msrvtt_pseudo_adverbs/ckpt_E_1000
```

## Citation

If you find this work helpful in your research, please cite:
```
@inproceedings{doughty2022how,
    author    = {Doughty, Hazel and Snoek, Cees G. M.},
    title     = {{H}ow {D}o {Y}ou {D}o {I}t? {F}ine-{G}rained {A}ction {U}nderstanding with {P}seudo-{A}dverbs},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```
