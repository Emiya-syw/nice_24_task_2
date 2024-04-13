export IMP_SILIENT_OTHERS=true

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


SPLIT="candidate_captions"

# merge eval
MODEL_CKPT="models/imp-v1-3b"
# MODEL_CKPT="imp-v1-3b" # eval your own checkpoint
EVAL_CKPT="${MODEL_CKPT//\//_}_1"
MODEL_PATH=$MODEL_CKPT
# MODEL_PATH="./checkpoints/$MODEL_CKPT" # eval your own checkpoint

for IDX in $(seq 0 $((CHUNKS-1))); do
    LOCAL_RANK=$IDX CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m imp_llava.eval.eval_nice_task_2 \
        --model-path $MODEL_PATH \
        --csv-file ./playground/data/nice_2/$SPLIT.csv \
        --image-folder ./playground/data/nice/images_20k  \
        --answers-file ./playground/data/nice_2/answers/$SPLIT/$EVAL_CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode phi2 &
done