#!/bin/bash
TIMEOUT=60
HOST=$(echo $1 | cut -d : -f 1)
PORT=$(echo $1 | cut -d : -f 2)

until nc -z $HOST $PORT 2>&1; do
    echo "Waiting for $HOST:$PORT..."
    sleep 1
    let TIMEOUT--
    if [ $TIMEOUT -eq 0 ]; then
        echo "Timeout reached waiting for $HOST:$PORT"
        exit 1
    fi
done

echo "$HOST:$PORT is available"