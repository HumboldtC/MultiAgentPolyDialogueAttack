CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model /beegfs/g3_hzq/icassp2026/model/Qwen3-30B-A3B-Instruct-2507 \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --port 8000
