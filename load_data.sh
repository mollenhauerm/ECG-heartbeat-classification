#!/bin/bash

DIR="data"

if [ "$(ls -A $DIR)" ]; then
echo "data already downloaded"
else
echo "downloading data"
kaggle datasets download -d shayanfazeli/heartbeat -p $DIR
fi
