# Usage Instructions for `distillation_with_zipformer.sh`

This document provides detailed instructions for running the `distillation_with_zipformer.sh` script, which performs knowledge distillation using the Zipformer model. The script is executed stage by stage, and the official model file `pretrained.pt` must be renamed and placed in the `exp_dir` directory. Refer to `RESULTS.md` in the same directory as the script for the source of the official model file.

## Prerequisites
- Rename the official `pretrained.pt` model file and place it in the `exp_dir` directory.
- Ensure the required dependencies and environment are set up as per the project requirements.
- The script assumes the presence of variables like `embedding_layer`, `num_codebook`, `exp_dir`, and `use_extracted_codebook`. Define these appropriately before running the script.
