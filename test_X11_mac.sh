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

xhost + 127.0.0.1

# Use Tensorflow backend
if [ $# -eq 0 ]; then
    docker run -v $PWD:/opt/TSML -e DISPLAY=docker.for.mac.host.internal:0 -v $PWD/data:/data -it tsml_test_container /bin/bash -c "cd /opt/TSML && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest"
else
    docker run -v $PWD:/opt/TSML -e DISPLAY=docker.for.mac.host.internal:0 -v $PWD/data:/data -it tsml_test_container /bin/bash -c "cd /opt/TSML && $ENV_VARS KERAS_BACKEND=tensorflow python3 -m unittest $@"
fi

