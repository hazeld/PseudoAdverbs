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

Code will be made available soon.
