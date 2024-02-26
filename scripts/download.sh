#!/bin/bash

cd "$DEFAULT_DATASET_PATH" || exit
wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/PS1.tar.gz
tar -xvf PS1.tar.gz

wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/PS2.tar.gz
tar -xvf PS2.tar.gz

wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/PS3.tar.gz
tar -xvf PS3.tar.gz

wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/PS4.tar.gz
tar -xvf PS4.tar.gz

wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/PS5.tar.gz
tar -xvf PS5.tar.gz

cd "$DEFAULT_CHECKPOINT_PATH" || exit
wget https://services.h2t.iar.kit.edu/webstash/publication/conference/ICRA/2024/bi_kvil/dcn.tar.gz
tar -xvf dcn.tar.gz
