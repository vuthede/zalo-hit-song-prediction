# Quick run to see the result
- Install necessary libs

- If you want to see analysis file to see some data exploration step. Look at `analysis.ipynb`

- `python train_and_test.py  --data csv/ --csv_metadata_path csv/metadata_embedded_in_mp3.csv --save_submission_file submission.csv`

- It take about 20-30 minutes, depends on specification of your laptop

# File Structure
`csv` dir: Contains some information Zalo supply, plus some generated features

`analysis.ipynb`: Data exploration

`train_and_test.py`. Aggerate features and use lightgbm model to train in 10 folds and get the average result.

`create_metadata_features.py`. Script to extract metadata features which are embedded in mp3 files.

`convert_mp3_to_wav.py`. Script toconvert mp3 files to wav files. **Note** Please use it with `create_audio_features.py` to generate some audio_features.

 - Run `convert_mp3_to_wav.py` first
 - Then run `create_audio_features.py` on generate wav files.
 
`create_audio_features.py`. Script to generate audio features.

`format_features.py`. Aggerate all features and put those into dataframe


`utils.py`. Some helper functions

# Some key ideas 
+ Using lightgbm model with 10 folds
+ Split dataset by album instead of by label (ranks)
+ Using target encoding using album, artist and ranks information. Baased on the fact the songs with the same album and artist tend to have the same rank.