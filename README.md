# Result in 2019 Zalo Hit Song Prediction AI Competition

2nd Place Public Leaderboard: 1.48740	RMSE
2nd Place Private Leaderboard: 1.48030	RMSE

# Quick run to see the result
- Install necessary libs

- To see some of our Exploratory Data Analysis step please see `analysis.ipynb`

- `python train_and_test.py  --data csv/ --csv_metadata_path csv/metadata_embedded_in_mp3.csv --save_submission_file submission.csv`

- It take about 20-30 minutes to run, depending on the specification of your machine

# File Structure

`csv` dir: Contains some information Zalo supply, plus some features we have generated

`analysis.ipynb`: Exploratory Data Analysis

`train_and_test.py`. Aggregate features and use lightgbm model to train on 10 folds, then average the result across the 10 models.

`create_metadata_features.py`. Script to extract metadata features which are embedded in mp3 files.

`convert_mp3_to_wav.py`. Script to convert mp3 files to wav files. **Note** Please use it with `create_audio_features.py` to generate some audio_features.

 - Run `convert_mp3_to_wav.py` first
 - Then run `create_audio_features.py` on generate wav files.
 
`create_audio_features.py`. Script to generate audio features.

`format_features.py`. Aggerate all features and put those into dataframe

`utils.py`. Some helper functions


# Some key ideas in our solution

+ Parsed mp3 metadata strings into various binary flags which seem to show relationship with rank, an example being isBeat whereby 'beat' tagged albums tends to result in worse rank.  
+ We realised that the release time for a given album often had the same corresponding second, and thus by grouping songs by release time we are able to produce a powerful feature describing the album
--> We used label encoding for a lot of these large categorical features. We discovered the artist id feature seemed to be encoding some information about how old the artist is (likely created in order of artist reaching Zalo). As, intuitively, very old artists are less likely to be top hits, and label encoding keeps this ordering, the decision tree-based model we were using could take advantage of this 'hidden' information.
+ Using average of predictions across 10 lightgbm models from 10 CV folds. We tried the maybe more conventional approach of regularising and training on 100% of the training data but found the model significantly overfit compared to 10 CV fold method.  
+ Split dataset by album instead of by label (ranks) then weight loss by rank to account for small class imbalance
+ Using target encoding using album, artist and ranks information. Based on the fact the songs with the same album and artist tend to have the same rank.

# Things we tried that didn't work out

+ Adding in features generated from short clip of the raw audio using the (very impressive!) essentia library and it's MusicExtractor function. Lightgbm mostly ignored these features relative to our final features
+ We made some models training exclusively on the lyrics (Hashed) and got RMSE around 3.5, then tried adding this model's rank predicton as a feature. Sometimes saw small improvement, but sometimes not, so ran out of time to incorporate it to final model.
+ Did extensive hyperparameter sweep using hyperopt but didn't get much improvement compared to manual tuning
+ We ran out of time to incorporate a kNN model, this should have help for cases where test data has duplicate (or almost duplicate) songs to the training data 
