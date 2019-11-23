#!/bin/bash
set -e

# Build test environment
echo -e "\n==============================="
echo -e  "|  Building test environment  |"
echo -e  "===============================\n"
    
if [[ "x$cache" == "xnocache" ]]; then
    docker build --no-cache -f containers/Ubuntu_18.04/Dockerfile ./ -t tsml_test_container
else
    docker build -f containers/Ubuntu_18.04/Dockerfile ./ -t tsml_test_container
fi
			
# Start testing
echo -e  "\n==============================="
echo -e  "|   Running tests             |"
echo -e  "===============================\n"

# Reduce verbosity and disable Python buffering
ENV_VARS="PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=on"

# Use Tensorflow backend
docker run -v $PWD:/opt/TSML -it tsml_test_container /bin/bash -c "cd /opt/TSML && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest discover"

# Use Theano backend
#docker run -it tsml_test_container /bin/bash -c "cd /opt/TSML && $ENV_VARS KERAS_BACKEND=theano python3 -m unittest discover"
