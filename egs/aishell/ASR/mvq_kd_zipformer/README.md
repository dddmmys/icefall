# Usage Instructions for `distillation_with_zipformer.sh`

This document provides detailed instructions for running the `distillation_with_zipformer.sh` script, which performs knowledge distillation using the Zipformer model. The script is executed stage by stage, and the official model file `pretrained.pt` must be renamed and placed in the `exp_dir` directory. Refer to `RESULTS.md` in the same directory as the script for the source of the official model file.

## Prerequisites
- Rename the official `pretrained.pt` model file and place it in the `exp_dir` directory.
- Ensure the required dependencies and environment are set up as per the project requirements.
- The script assumes the presence of variables like `embedding_layer`, `num_codebook`, `exp_dir`, and `use_extracted_codebook`. Define these appropriately before running the script.

## Stage-by-Stage Execution
The script is divided into multiple stages, each performing a specific task in the knowledge distillation process. Below are the details for each stage.

### Stage 2: Extract Codebook Indices for 3 Teachers (6 Layers Total)
This stage extracts codebook indices for three teacher models (`zipformer_s_55`, `zipformer_l_56`, `zipformer_m_55`), each with two embedding layers.

#### Extract Codebook Indices for `zipformer_s_55`
```bash
export MODEL_TYPE="zipformer_s_55"
teacher_model_id=zipformer_s_55
```

- **First Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[0]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[0]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 192 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

- **Second Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[1]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[1]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 256 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

#### Extract Codebook Indices for `zipformer_l_56`
```bash
export MODEL_TYPE="zipformer_l_56"
teacher_model_id=zipformer_l_56
```

- **First Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[0]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[0]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 192 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

- **Second Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[1]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[1]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 768 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

#### Extract Codebook Indices for `zipformer_m_55`
```bash
export MODEL_TYPE="zipformer_m_55"
teacher_model_id=zipformer_m_55
```

- **First Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[0]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[0]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 192 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

- **Second Layer**
```bash
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    ./mvq_kd_zipformer/extract_codebook_index.py \
      --kd-exp-dir $exp_dir \
      --embedding-layer ${embedding_layer[1]} \
      --num-utts 2000 \
      --num-codebooks ${num_codebook[1]} \
      --max-duration 300 \
      --teacher-model-id $teacher_model_id \
      --spec-aug-time-warp-factor -1 \
      --embedding-dim 512 \
      --num-utts-extracted 360294 \
      --use-extracted-codebook $use_extracted_codebook
fi
```

### Stage 3: Aggregate Codebook Indices
This stage aggregates codebook indices across layers and teachers into a single file.

#### Aggregate Layer-wise Indices for `zipformer_s_55`
```bash
teacher_model_id=zipformer_s_55
use_mul_tea=False
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  ./mvq_kd_zipformer/combine_jsonl.py \
    --output-path $exp_dir \
    --teacher-model-id $teacher_model_id \
    --use-mul-tea $use_mul_tea
fi
```

#### Aggregate Layer-wise Indices for `zipformer_l_56`
```bash
teacher_model_id=zipformer_l_56
use_mul_tea=False
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  ./mvq_kd_zipformer/combine_jsonl.py \
    --output-path $exp_dir \
    --teacher-model-id $teacher_model_id \
    --use-mul-tea $use_mul_tea
fi
```

#### Aggregate Layer-wise Indices for `zipformer_m_55`
```bash
teacher_model_id=zipformer_m_55
use_mul_tea=False
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  ./mvq_kd_zipformer/combine_jsonl.py \
    --output-path $exp_dir \
    --teacher-model-id $teacher_model_id \
    --use-mul-tea $use_mul_tea
fi
```

#### Aggregate Indices for All 3 Teachers (6 Layers)
```bash
use_mul_tea=True
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  ./mvq_kd_zipformer/combine_jsonl.py \
    --output-path $exp_dir \
    --teacher-model-id $teacher_model_id \
    --use-mul-tea $use_mul_tea
fi
```

### Stage 4: Extract Pruned RNNT Loss Values
This stage extracts loss values for the teacher models.

#### Extract Loss for `zipformer_s_55`
- Create a symbolic link for `zipformer_s_55.pt` and rename it to `epoch-55.pt`.
```bash
teacher_model_id=zipformer_s_55
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ./mvq_kd_zipformer/train_s_confidence.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --num-epochs 56 \
    --start-epoch 56 \
    --max-duration 300 \
    --exp-dir $exp_dir \
    --lang-dir data/lang_char \
    --context-size 1 \
    --teacher-model-id $teacher_model_id
fi
```

#### Extract Loss for `zipformer_m_55`
- Create a symbolic link for `zipformer_s_55.pt` in `$exp_dir/zipformer_m` and rename it to `epoch-55.pt`.
```bash
teacher_model_id=zipformer_m_55
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ./mvq_kd_zipformer/train_m_confidence.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --num-epochs 56 \
    --start-epoch 56 \
    --max-duration 300 \
    --exp-dir $exp_dir/zipformer_m \
    --lang-dir data/lang_char \
    --context-size 1 \
    --teacher-model-id $teacher_model_id
fi
```

#### Extract Loss for `zipformer_l_56`
- Create a symbolic link for `zipformer_l_56.pt` and rename it to `epoch-56.pt`.
```bash
teacher_model_id=zipformer_l_56
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ./mvq_kd_zipformer/train_l_confidence.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --num-epochs 57 \
    --start-epoch 57 \
    --max-duration 300 \
    --exp-dir $exp_dir \
    --lang-dir data/lang_char \
    --context-size 1 \
    --teacher-model-id $teacher_model_id
fi
```

#### Aggregate Loss Values for All 3 Teachers
```bash
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  ./mvq_kd_zipformer/combine_logs.py \
    --output-path $exp_dir
fi
```

### Stage 5: Train Student Model `zipformer-xs`
This stage trains the student model `zipformer-xs` using different fusion strategies.

#### Train with Full Dataset and Sample-level Confidence Fusion
```bash
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  WORLD_SIZE=$(echo ${CUDA_VISIBLE_DEVICES} | awk '{n=split($1, _, ","); print n}')
  ./mvq_kd_zipformer/train_xs_new.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --spec-aug-time-warp-factor -1 \
    --max-duration 300 \
    --world-size ${WORLD_SIZE} \
    --num-epochs 60 \
    --start-epoch 1 \
    --save-every-n 100000 \
    --exp-dir $exp_dir/student/xs_full_dataset/confidence \
    --distillation-layer $embedding_layers \
    --num-codebooks $num_codebooks \
    --enable-distillation True \
    --enable-multilayer-distillation True \
    --use-mul-tea use_mul_tea \
    --codebook-loss-scale 1.0 \
    --layers-weight-opt "uncertainty2" \
    --tea-weight-opt "confidence" \
    --data-fraction 1
fi
```

#### Train with Full Dataset and Average Fusion
```bash
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  WORLD_SIZE=$(echo ${CUDA_VISIBLE_DEVICES} | awk '{n=split($1, _, ","); print n}')
  ./mvq_kd_zipformer/train_xs_new.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --spec-aug-time-warp-factor -1 \
    --max-duration 300 \
    --world-size ${WORLD_SIZE} \
    --num-epochs 60 \
    --start-epoch 1 \
    --save-every-n 100000 \
    --exp-dir $exp_dir/student/xs_full_dataset/avg \
    --distillation-layer $embedding_layers \
    --num-codebooks $num_codebooks \
    --enable-distillation True \
    --enable-multilayer-distillation True \
    --use-mul-tea use_mul_tea \
    --codebook-loss-scale 1.0 \
    --layers-weight-opt "uncertainty2" \
    --tea-weight-opt "avg" \
    --data-fraction 1
fi
```

#### Train with Full Dataset, Confidence Fusion, and Early Stopping at Epoch 50
```bash
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  WORLD_SIZE=$(echo ${CUDA_VISIBLE_DEVICES} | awk '{n=split($1, _, ","); print n}')
  ./mvq_kd_zipformer/train_xs_new.py \
    --manifest-dir $exp_dir/combined_${embedding_layer[0]}_${embedding_layer[1]} \
    --spec-aug-time-warp-factor -1 \
    --max-duration 300 \
    --world-size ${WORLD_SIZE} \
    --num-epochs 60 \
    --start-epoch 51 \
    --save-every-n 100000 \
    --exp-dir $exp_dir/student/xs_full_dataset/confidence/early-stop-50 \
    --distillation-layer $embedding_layers \
    --num-codebooks $num_codebooks \
    --enable-distillation False \
    --enable-multilayer-distillation True \
    --use-mul-tea use_mul_tea \
    --codebook-loss-scale 1.0 \
    --layers-weight-opt "uncertainty2" \
    --tea-weight-opt "confidence" \
    --data-fraction 1 \
    --early-stop True
fi
```

### Stage 6: Decode Student Model `zipformer-xs`
This stage decodes the trained student model using different configurations.

#### Decode Distilled Model
```bash
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  ./mvq_kd_zipformer/decode_xs.py \
    --decoding-method "greedy_search" \
    --epoch 60 \
    --avg 10 \
    --max-duration 750 \
    --exp-dir $exp_dir/student/xs_full_dataset/confidence \
    --lang-dir data/lang_char \
    --context-size 1 \
    --distillation-layer $embedding_layers \
    --num-codebooks $num_codebooks \
    --layers-weight-opt "uncertainty2" \
    --enable-distillation True \
    --use-averaged-model True
fi
```

#### Decode Non-Distilled Model
```bash
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  ./mvq_kd_zipformer/decode_xs.py \
    --decoding-method "greedy_search" \
    --epoch 60 \
    --avg 10 \
    --max-duration 750 \
    --exp-dir $exp_dir/student/xs_full_dataset/confidence/early-stop-50 \
    --lang-dir data/lang_char \
    --context-size 1 \
    --distillation-layer $embedding_layers \
    --num-codebooks $num_codebooks \
    --layers-weight-opt "uncertainty2" \
    --enable-distillation False \
    --use-averaged-model True
fi
```

## Notes
- Ensure that `CUDA_VISIBLE_DEVICES` is set to determine `WORLD_SIZE` for distributed training.
- The `embedding_layers` and `num_codebooks` variables must be defined appropriately before running the script.
- Check `RESULTS.md` in the script's directory for details on obtaining the official `pretrained.pt` model file.
- The script assumes the existence of the `mvq_kd_zipformer` directory with the necessary Python scripts (`extract_codebook_index.py`, `combine_jsonl.py`, `train_s_confidence.py`, `train_m_confidence.py`, `train_l_confidence.py`, `combine_logs.py`, `train_xs_new.py`, `decode_xs.py`).