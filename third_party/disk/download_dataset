#/usr/bin/env bash

# get the data (zipped)
wget -r https://datasets.epfl.ch/disk-data/index.html

cd datasets.epfl.ch/disk-data;

# check for MD5 match
md5sum -c md5sum.txt;
if [ $? ]; then
    echo "MD5 mismatch (corrupt download)";
    return 1;
fi

# create a crude progress counter
ITER=1;
TOTAL=138;
# unzip test scenes
cd imw2020-val/scenes;
for SCENE_TAR in *.tar.gz; do
    echo "Unzipping $SCENE_TAR ($ITER / $TOTAL)";
    tar -xz --strip-components=3 -f $SCENE_TAR;
    rm $SCENE_TAR;
    ITER=$(($ITER+1));
done

# unzip megadepth scenes
cd ../../megadepth/scenes;
for SCENE_TAR in *.tar; do
    echo "Unzipping $SCENE_TAR ($ITER / $TOTAL)";
    tar -x --strip-components=3 -f $SCENE_TAR;
    rm $SCENE_TAR;
    ITER=$(($ITER+1));
done

# create symlinks as train and test datasets
cd ../../
ln -s megadepth train
ln -s imw2020-val test
