#!/bin/bash

RECIPES=(
    "release/ecoscope-workflows-ext-gamm-trend-analysis"
)
project_root=$(pwd)/src/ecoscope-workflows-ext-gamm-trend-analysis
export HATCH_VCS_VERSION=$(cd $project_root && hatch version)
echo "HATCH_VCS_VERSION=$HATCH_VCS_VERSION"


echo "Building recipes: ${RECIPES[@]}"

pixi clean cache --yes

rm -rf /tmp/ecoscope-workflows-custom/release/artifacts
mkdir -p /tmp/ecoscope-workflows-custom/release/artifacts

for rec in "${RECIPES[@]}"; do
    echo "Building $rec"
    rattler-build build \
    --recipe $(pwd)/publish/recipes/${rec}.yaml \
    --output-dir /tmp/ecoscope-workflows-custom/release/artifacts \
    --channel https://prefix.dev/ecoscope-workflows \
    --channel conda-forge
done
