# Create metadata
python3 /model/create_metadata_features.py  --in_mp3_dirs /data/train   /data/private --out_csv_metadata_path /model/metadata.csv

# Train and predict and save predictions into .csvd file
python3 /model/train_and_test.py --data  /data/info  --csv_metadata_path /model/metadata.csv --save_submission_file /result/submission.csv