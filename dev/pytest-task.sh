#!/bin/bash

python_version=$1

command="pixi run \
--manifest-path src/ecoscope-workflows-ext-gamm-trend-analysis/pyproject.toml \
--environment test-py${python_version} \
pytest src/ecoscope-workflows-ext-gamm-trend-analysis/tests -vv"

shift 1
if [ -n "$*" ]; then
    extra_args=$*
    command="$command $extra_args"
fi

echo "Running command: $command"
eval $command
