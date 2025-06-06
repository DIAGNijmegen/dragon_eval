#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t joeranbosma/dragon_eval:latest -t joeranbosma/dragon_eval:v0.2.11 "$SCRIPTPATH"
