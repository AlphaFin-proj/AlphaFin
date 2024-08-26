cd src

chatglm_path="/path/to/chatglm2_6b"
stockgpt2_path="/path/to/stockgpt_stage2_lora"
embedding_path="/path/to/BGE-Large"

python stage2_financial_qa/webui/run.py \
    --base_model_path ${chatglm_path} \
    --lora_ckpt_path ${stockgpt2_path} \
    --embedding_model_path ${embedding_path}