#!/bin/bash
# Samples the swarm's mixtures with olmix, in the eval venv ($OLMIX_PYTHON).
#
# Usage (from the repository root, on the Roihu-GPU login node):
#   ./generate_mixes.sh configs/<experiment>.yaml
set -e

source env.sh

config_file="${1:?usage: $0 configs/<experiment>.yaml}"

"$OLMIX_PYTHON" scripts/generate_variants.py "$config_file"

echo "Ready to launch!"
