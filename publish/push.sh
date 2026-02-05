#!/bin/bash

for file in /tmp/ecoscope-workflows-custom/release/artifacts/**/*.conda; do
    rattler-build upload prefix -c ecoscope-workflows-custom "$file"
done
