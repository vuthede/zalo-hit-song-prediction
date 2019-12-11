import os
import glob
import ntpath
import pandas as pd
import eyed3
from mp3_tagger import MP3File, VERSION_1, VERSION_2, VERSION_BOTH
import argparse

parser = argparse.ArgumentParser(description='Script to generate metadata features from mp3 files')
parser.add_argument('--in_mp3_dirs', nargs='+', required=True, help='Video directories that contains mp3 files')
parser.add_argument('--out_csv_metadata_path', type=str, required=True, help='.csv file contain generated metadata features')




'''
 \brief Get metadata of a mp3 file given the path to the file
'''
def get_track_info(file):
    try:
        info = {}
        audiofile = eyed3.load(file)
        mp3 = MP3File(file)
        mp3.set_version(VERSION_2)
        mp3.save()
        # Get all tags.
        copyright = mp3.copyright
        if audiofile is not None:
            info["album"] = audiofile.tag.album
            if (audiofile.tag.genre):
                info["genre"] = audiofile.tag.genre.name
            else:
                info["genre"] = None
            info["album_artist"] = audiofile.tag.album_artist
            info["track"] = audiofile.tag.track_num
            info["lyric"] = "".join([i.text for i in audiofile.tag.lyrics])
            info["length"]= audiofile.info.time_secs
            if len(copyright):
                info["copyright"] = copyright
            else:
                info["copyright"] = None
            return info
        else:
            info["album"] = None
            info["genre"] = None
            info["album_artist"] = None
            info["track"] = None
            info["lyric"] = None
            info["length"]= 0
            if len(copyright):
                info["copyright"] = copyright
            else:
                info["copyright"] = None
            return info
            
    except:
        print("Somthing wrong with file: {}".format(file))
        info["album"] = None
        info["genre"] = None
        info["album_artist"] = None
        info["track"] = None
        info["lyric"] = None
        info["length"]= 0
        if len(copyright):
            info["copyright"] = None
        else:
            info["copyright"] = None
        return info

'''
\ brief Get all track info of all mp3 songs in list of directory given. 
'''
def get_all_track_info(list_dir, out_csv):
    IDs = []
    infos = {"album":[], "genre":[], "album_artist":[], "track":[], "lyric":[],"length":[], "copyright":[]}
    
    files = []
    for directory in list_dir:
        files += glob.glob(directory + "/*.mp3")
    
    IDs = [ntpath.basename(f).replace(".mp3", "") for f in files]
    for f in files:
        info = get_track_info(f)
        for k,v in info.items():
            infos[k].append(v)
        
    df = pd.DataFrame(data={"ID":IDs, "album":infos["album"], 
                            "genre":infos["genre"], "album_artist":infos["album_artist"], 
                            "track":infos["track"],"lyric":infos["lyric"], "length":infos["length"],
                            "copyright":infos["copyright"]})
    
    # df.to_csv("../csv/all_track_info.csv")
    df.to_csv(out_csv)


if __name__=="__main__":
    # Parse params
    args = parser.parse_args()
    IN_MP3_DIRECTORIES = args.in_mp3_dirs
    OUT_CSV_METADATA_PATH= args.out_csv_metadata_path

    # Check directories exist
    for directory in IN_MP3_DIRECTORIES:
        assert os.path.isdir(directory),  f"Directory {directory} doesnot exist!!!"

    # Get metadata info and write to csv
    get_all_track_info(list_dir=IN_MP3_DIRECTORIES, out_csv=OUT_CSV_METADATA_PATH)