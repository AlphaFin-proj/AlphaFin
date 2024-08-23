cd src

python stage2_financial_qa/webui/run.py \
    --base_model_path /path/to/chatglm2_6b \
    --lora_ckpt_path /path/to/stockgpt_lora \
    --embedding_model_path /path/to/BGE-Large