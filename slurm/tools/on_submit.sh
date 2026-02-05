#!/usr/bin/env bash
set -euo pipefail
ssh -o BatchMode=yes aisurrey-submit01.surrey.ac.uk "$@"
