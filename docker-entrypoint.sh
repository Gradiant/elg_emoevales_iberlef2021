#!/bin/sh

exec /elg -- venv/bin/gunicorn --bind=0.0.0.0:8866 "--workers=$WORKERS" --worker-tmp-dir=/dev/shm "$@" serve:app
