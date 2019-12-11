"""
Note: This script is based mostly on this script
https://github.com/tiepvupsu/zalo_voice/blob/master/preprocess.py
"""
import os, subprocess

import argparse

parser = argparse.ArgumentParser(description='Script to convert mp3 files to wav files')
parser.add_argument('--in_mp3_dir', required=True, help='Video directory that contains mp3 files')
parser.add_argument('--out_wav_dir', type=str, required=True, help='Output directory contains wav files')
parser.add_argument('--frame_rate', type=float, default=44100, help='Framerate of converted wav files')
parser.add_argument('--second_start', type=float, default=20, help='From what second to cut the audio fiile')
parser.add_argument('--duration', type=float, default=20, help='How many seconds want to cut')


args = parser.parse_args()

INPUT_FOLDER = args.in_mp3_dir
OUTPUT_FOLDER = args.out_wav_dir
CONVERT_RATE = float(args.frame_rate)
START_SECOND = float(args.second_start)
DURATION = float(args.duration)

def convert_and_sample(src_folder, dst_folder):
    if not os.path.isdir(src_folder):
        print("Source folder doesn't exist, continue")
    if not os.path.isdir(dst_folder): 
        os.makedirs(dst_folder)
        
    fns = [fn for fn in os.listdir(src_folder) if 
                any(map(fn.endswith, ['.mp3', '.wav', '.amr']))]

    for i, fn in enumerate(fns): 
        old_fn = os.path.join(src_folder, fn) 
        new_fn = os.path.join(dst_folder, fn + '.wav')
        if os.path.isfile(new_fn): 
            continue
            
        # convert all file to wav
        subprocess.call(['ffmpeg', '-loglevel', 'panic', '-i', old_fn,
                         '-ss', START_SECOND, '-t', DURATION, 
                         '-ar', CONVERT_RATE, '-c', 'copy', new_fn])
        if (i+1)%100 == 0:
            print('{}/{}: {}'.format(i+1, len(fns), new_fn))
            
convert_and_sample(INPUT_FOLDER, OUTPUT_FOLDER)