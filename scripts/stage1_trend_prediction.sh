export HF_ENDPOINT=https://hf-mirror.com

cd src

tushare_token="your_tushare_token"

chatglm_path="/path/to/chatglm2_6b"
stockgpt1_path="/path/to/stockgpt_stage1_lora"

testdata_path="data/stage1_testdata.json"
output_path="../outputs"

stockgpt_pred_jsonl="stockgpt_prediction.jsonl"
mldl_pred_xlsx="mldl_prediction.xlsx"
stockgpt_mldl_pred_xlsx="stockgpt_mldl.xlsx"
final_name="strategy_result"

# 1. Preparetion: download kline db data
mkdir -p db_file
huggingface-cli download --resume-download --local-dir-use-symlinks False AlphaFin/stage1_db_file --local-dir ./db_file --repo-type dataset

# 2. Inference: stock trend prediction
python stage1_trend_prediction/stockgpt_inf.py \
    --model_name_or_path ${chatglm_path} \
    --lora_name_or_path ${stockgpt1_path} \
    --data_path ${testdata_path} \
    --output_path ${output_path}/${stockgpt_pred_jsonl}

# 3. PostProcess: handle invalid values
python stage1_trend_prediction/dataprocess_stockgpt.py \
    --stockgpt_pred_path ${output_path}/${stockgpt_pred_jsonl} \
    --mldl_pred_path ${output_path}/${mldl_pred_xlsx} \
    --save_path ${output_path}/${stockgpt_mldl_pred_xlsx}

# 4. Execute: strategy simulation
python stage1_trend_prediction/test_strategy.py \
    --tushare_token ${tushare_token} \
    --stockgpt_mldl_path ${output_path}/${stockgpt_mldl_pred_xlsx} \
    --save_dir ${output_path}/${final_name} \
    --file_name ${final_name} \
    |& tee ${output_path}/strategy_test.log