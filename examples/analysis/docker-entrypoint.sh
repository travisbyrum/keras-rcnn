#!/bin/sh -e

PROJECT_NAME=bco-analysis
WORKING_DIR=/opt/bco

case "$1" in
train)
  shift
  exec python3 "$WORKING_DIR/training.py" $@
  ;;
validate)
  shift
  exec python3 "$WORKING_DIR/validation.py" $@
  ;;
*)
  echo "command not found"
  exit 1
  ;;
esac
