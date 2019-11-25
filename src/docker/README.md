# Struture folder in ZALO server
The /data folder is organized as followed: MP3 files in 3 sub directors: "/data/train/*.mp3", "/data/test/*.mp3", "/data/private/*.mp3", meta data in "/data/info/train_info.tsv", "/data/info/train_rank.csv", "/data/info/test_info.tsv", "/data/info/private_info.tsv".

# Before run the docker
Structure the data in local machine similar to how Zalo structured which describe above

# How to run this docker
sudo docker run -v `<path-to-local-dir-zalo-data>`:/data -v `<path-to-local-dir-for-saving-submisison-file>`:/result `<name-of-docker-image>` /bin/bash /model/predict.sh

Note: This docker will:
+ Running a python script to generate metadata.csv first
+ Running another python  script to merge dataframe, creating features, cast type of features, train lightgbm in 10 folds, save model and make prediction. Finally save the predictions to a .csv file.

# How to edit the existing docker image to submit in future

1> Some useful docker command
- Showing docker images : `sudo docker images ls`
- Showing docker containers : `sudo docker ps -a`
- Copy data from local machine to docker: `sudo docker cp <path-to-dir-or-file-in-local-machine> <CONTAINER_ID>:/<path-in-docker>`
- Save a docker container as an docker image: `sudo docker commit <CONTAINER_ID> <new-docker-image-name>:<TAG>`
- See checksum of docker image : `docker images --digests`

2> For example what I did in my local machine
- For example all the code are ready to put into docker, then using `docker cp` command to copy code from local machine to docker. To be consitent, we copy to `/model` directory inside docker. And all the files should be on the same directory `/models` and there is no sub-folder. For instance the command I used in my local machine:

`sudo docker cp /home/vuthede/AI/zalo/zalo-hit-song-prediction/src/docker/* <dbfe16c99f11>:/model`

- After copy new code to the container instane of the docker image. Then we will save it and create an new docker image. For example in my local

`sudo docker commit dbfe16c99f11 zalo_hitsong_prediction:baseline4 `

- Export docker image

 `docker save –o [output file] [image id]`