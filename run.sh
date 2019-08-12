# !/bin/bash

# EXP AND PATH SETTING 
. ./config.sh

# Step 1: Audio2vec
bash src/audio2vec.sh

# Step 2: Clustering (K-means)
bash src/cluster.sh

# Step 3: Learning the mapping relationship (GAN)
bash src/mapping.sh
