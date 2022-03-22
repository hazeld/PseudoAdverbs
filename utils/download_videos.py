import os
import argparse
import subprocess
import pandas as pd
import tqdm

parser = argparse.ArgumentParser('Scripts to download videos for VATEX Adverbs, MSR-VTT Adverbs and ActivityNet Adverbs')
parser.add_argument('annotation_file', type=str)
parser.add_argument('output_dir', type=str)
args = parser.parse_args()

def read_annotations(filename):
    return pd.read_csv(filename)

def trim_vid(row, output_dir):
    vid_path = os.path.join(output_dir, str(row['clip_id']) + '.mp4')
    if not os.path.isfile(vid_path):
        print('Video ' + str(row['clip_id']) + ' not present, skipping')
        return
    original_vid_path = vid_path.replace('.mp4', '_original.mp4')
    os.rename(vid_path, original_vid_path)
    start_seconds = row['start_time']
    duration = row['end_time'] - start_seconds
    subprocess.call(['ffmpeg', '-ss', str(start_seconds), '-i', original_vid_path, '-t', str(duration), '-c:a', 'ac3', '-c:v', 'libx264', '--', vid_path])
    os.remove(original_vid_path)

def download_vid(row, output_dir):
    try:
        subprocess.check_output(['youtube-dl', '-f', 'mp4', '-i', '-o', os.path.join(output_dir, str(row['clip_id'])+'.%(ext)s'), '--', row['youtube_id']])
    except subprocess.CalledProcessError:
        return True
    return False

if __name__ == '__main__':
    video_list_df = read_annotations(args.annotation_file)
    clips_with_errors = []
    for i, row in tqdm.tqdm(video_list_df.iterrows(), total=video_list_df.shape[0]):
        error = download_vid(row, args.output_dir)
        if error:
            clips_with_errors.append(row['clip_id'])
        trim_vid(row, args.output_dir)
    print('Unable download the following videos: ')
    print(clips_with_errors)
