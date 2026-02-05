#!/bin/bash

workflow=$(echo $1 | tr '_' '-')
api=$2
mode=$3
flags="${@:4}"

manifest_path=workflows/${workflow}/ecoscope-workflows-${workflow}-workflow/pixi.toml
pixi_task=test-${api}-${mode}-mock-io

# update workflow env because ecoscope-workflows-* versions may have
# changed if `build-release` was invoked since last lockfile update
pixi update --manifest-path $manifest_path

# run the test
pixi run --manifest-path $manifest_path --locked -e test $pixi_task \
    --case all-grouper \
    $flags
