#!/bin/bash
set -e

# The first argument is the name of the venv
VENV_NAME=$1
# The remainder are passed to the python executable
ARGS=${@:2}

UNAME=`uname -m`

if [ "$UNAME" == "aarch64" ]; then
    # For M1, preload libgomp to avoid this bug https://stackoverflow.com/questions/67735216/after-using-pip-i-get-the-error-scikit-learn-has-not-been-built-correctly
    LD_PRELOAD="libgomp.so.1 /app/keras/.venv/lib/python3.8/site-packages/akida.libs/libgomp-01527a09.so.1.0.0" /app/$VENV_NAME/.venv/bin/python3 -u $ARGS
else
    # Run using python executable from the provided venv
    /app/$VENV_NAME/.venv/bin/python3 -u $ARGS
fi
