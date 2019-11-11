import os, subprocess

INPUT_FOLDER = "/data/zalo/hit-song-prediction/test"
OUTPUT_FOLDER = "/data/zalo/hit-song-prediction/test-image-samples"
CONVERT_RATE = "44100.0"
START_SECOND = "20"
DURATION = "20"

def get_image_of_mp3_file(src_folder, dst_folder):
    if not os.path.isdir(src_folder):
        print("Source folder doesn't exist, continue")
    if not os.path.isdir(dst_folder): 
        os.makedirs(dst_folder)
    
    
    fns = [fn for fn in os.listdir(src_folder) if 
                any(map(fn.endswith, ['.mp3', '.wav', '.amr']))]
    
    for i, fn in enumerate(fns): 
        old_fn = os.path.join(src_folder, fn) 
        new_fn = os.path.join(dst_folder, fn + '.jpg')
        if os.path.isfile(new_fn): 
            print("hahahaha")
            continue
        subprocess.call(["ffmpeg", "-i", old_fn, new_fn])
        if (i+1)%100 == 0:
            print('{}/{}: {}'.format(i+1, len(fns), new_fn))
            
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
        # new_fn = dst_folder + fn + '.wav'
        if os.path.isfile(new_fn): 
            print("hahahaha")
            continue
        # convert all file to wav, mono, sample rate 8000
        subprocess.call(['ffmpeg', '-loglevel', 'panic', '-i', old_fn,
                         '-ss', START_SECOND, '-t', DURATION, 
                         '-ar', CONVERT_RATE, '-c', 'copy', new_fn])
        if (i+1)%100 == 0:
            print('{}/{}: {}'.format(i+1, len(fns), new_fn))

# convert_and_sample(INPUT_FOLDER, OUTPUT_FOLDER)
    
get_image_of_mp3_file(INPUT_FOLDER, OUTPUT_FOLDER)