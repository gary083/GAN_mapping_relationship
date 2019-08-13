#!/bin/bash

. ./config.sh
. ./path.sh

# Process TIMIT
bash src/process-timit/process_timit.sh $root_path $timit_path

