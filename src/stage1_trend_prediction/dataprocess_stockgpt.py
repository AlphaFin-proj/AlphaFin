import json
import pandas as pd
from fire import Fire

def getGtUpDown(gt):
    """
    从标准答案中提取涨跌结果

    Args:
        gt: str
    Returns:
        up_down: "涨"/"跌"，若无法提取出答案，返回 None

    """
    if '涨' in gt:
        return '涨'
    elif '跌' in gt:
        return '跌'
    else:
        return None

def getPredUpDownStrict(pred):
    """
    从模型预测结果中提取涨跌结果（严格）

    Args:
        pred: str
    Returns:
        up_down: -1/0/1, -1表示跌，1表示涨，0表示无法提取出结果

    """
    keywords_up = ["涨", "上涨", "底部", "拉升", "向上", "大涨", "看好", "向好", "上升", "金叉", "持有", "乐观", "突破"]
    keywords_down = ["顶部", "不建议", "然而", "但是", "跌", "下跌", "弱", "下降", "向下", "卖出", "大跌", "看衰", "向差", "暴跌", "死叉", "波动", "暂不推荐", "风险", "谨慎", "悲观"]

    if '因此' in pred or '综上所述' in pred:
        keyword = '因此' if '因此' in pred else '综上所述'

        start_index = pred.find(keyword)
        pred = pred[start_index:]

        for b, keyword in enumerate(keywords_up + keywords_down):
            if keyword in pred:
                if b < len(keywords_up):
                    return 1
                else:
                    return -1
    else:
        return 0

def getPredUpDownLoose(pred):
    """
    从模型预测结果中提取涨跌结果（宽松）

    Args:
        pred: str
    Returns:
        up_down: -1/0/1, -1表示跌，1表示涨，0表示无法提取出结果

    """
    keywords_up = ["涨", "上涨", "底部","拉升","向上","大涨","看好","向好","上升","金叉","持有","乐观","突破"]
    keywords_down = ["顶部","不建议","然而","但是","跌", "下跌", "弱","下降","向下","卖出","大跌","看衰","向差","暴跌","死叉","波动","暂不推荐","风险","谨慎","悲观"]

    if '因此' in pred:
        start_index = pred.find("因此")
        pred_answer = pred[start_index:]
        for b, keyword in enumerate(keywords_up + keywords_down):
            if keyword in pred_answer:
                if b < len(keywords_up):
                    return 1          
                else:
                    return -1
    elif '综上所述' in pred:
        start_index = pred.find("综上所述")
        pred_answer = pred[start_index:]
        for b, keyword in enumerate(keywords_up + keywords_down):
            if keyword in pred_answer:
                if b < len(keywords_up):
                    return 1                
                else:
                    return -1
    elif '最终收益结果是' in pred:
        start_index = pred.find("最终收益结果是")
        pred_answer = pred[start_index:]
        for b, keyword in enumerate(keywords_up + keywords_down):
            if keyword in pred_answer:
                if b < len(keywords_up):
                    return 1
                else:
                    return -1
    else:
        if '涨' in pred:
            return 1
        elif '跌' in pred:
            return -1
        else:
            return 0

def changeKey(d, old_k, new_k):
    d[new_k] = d.pop(old_k)
    return d

def main(stockgpt_pred_path, mldl_pred_path, save_path):
    # 读取 StockGPT 在1000test上的推理结果 json 文件
    with open(stockgpt_pred_path, 'r') as f:
        json_data = [json.loads(x) for x in f.readlines()]
    json_data = [changeKey(x, 'output', 'ground_truth') for x in json_data]

    target_models = ['StockGPT']

    parsed_data = []
    # 基于规则方法提取目标模型的预测结果
    for sample in json_data:
        temp = {
            'stock_name': sample['stock_name'],
            'stock_code': sample['stock_code'],
            'date': sample['date'],
        }
        temp['ground_truth'] = getGtUpDown(sample['ground_truth'])

        for model_name in target_models:
            model_answer = sample[model_name]
            temp[model_name] = getPredUpDownStrict(model_answer)

        parsed_data.append(temp)

    # 计算无效答案比例
    pred_df = pd.DataFrame(parsed_data)
    pred_df = pred_df.sort_values(by=['date'])
    pred_df = pred_df.dropna(subset=['date'])
    pred_df = pred_df.fillna(0)
    pred_df = pred_df.reset_index(drop=True)
    print('pred_df:\n', pred_df)

    zero_counts = (pred_df == 0).sum() 
    total_samples = len(pred_df)  
    zero_percentages = zero_counts / total_samples  
    print("无效答案比例: ", zero_percentages)


    # 用 ML&DL 的预测结果替代 StockGPT 的无效答案
    df_stockgpt = pred_df
    df_all = pd.read_excel(mldl_pred_path, engine='openpyxl')  # stock_name, stock_code, date, ground_truth, RNN, ALSTM, DecisionTree.xlsx, Logistic, SVM, Transformers, gru, lstm, randomforest, xgboost

    df_stockgpt['date'] = df_stockgpt['date'].apply(lambda x:x.split()[0]).astype(str)
    df_all['date'] = df_all['date'].astype(str)

    # index 改为 <stock_name>_<date> 格式
    df_stockgpt.index = df_stockgpt['stock_name'] + '_' + df_stockgpt['date']
    df_all.index = df_all['stock_name'] + '_' + df_all['date']

    # 按照 index 进行去重
    df_stockgpt = df_stockgpt[~df_stockgpt.index.duplicated()]
    df_all = df_all[~df_all.index.duplicated()]

    # 两表格取并集
    common_index = set(df_stockgpt.index) & set(df_all.index)
    df_stockgpt = df_stockgpt[df_stockgpt.index.isin(common_index)]
    df_all = df_all[df_all.index.isin(common_index)]

    df_stockgpt = df_stockgpt.sort_index()
    df_all = df_all.sort_index()

    df_stockgpt = df_stockgpt.sort_values(by=['stock_name'])
    df_stockgpt = df_stockgpt.sort_values(by=['date'])

    df_all = df_all.sort_values(by=['stock_name'])
    df_all = df_all.sort_values(by=['date'])

    # 将StockGPT结果拷贝到ML&DL表格中
    df_all['StockGPT'] = df_stockgpt['StockGPT']


    df_all = df_all.rename(columns={'DecisionTree.xlsx':'DecisionTree'})

    # 1代表买入，-1代表卖出，将0全部替换为-1
    df_all['ground_truth'] = df_all['ground_truth'].replace({1: 1, 0: -1})
    df_all['lstm'] = df_all['lstm'].replace({1:1, 0:-1})
    df_all['RNN'] = df_all['RNN'].replace({1:1, 0:-1})
    df_all['gru'] = df_all['gru'].replace({1:1, 0:-1})
    df_all['ALSTM'] = df_all['ALSTM'].replace({1:1, 0:-1})
    df_all['DecisionTree'] = df_all['DecisionTree'].replace({1:1, 0:-1})

    df_all['Logistic'] = df_all['Logistic'].replace({1:1, 0:-1})
    df_all['xgboost'] = df_all['xgboost'].replace({1:1, 0:-1})
    df_all['Transformers'] = df_all['Transformers'].replace({1:1, 0:-1})
    df_all['SVM'] = df_all['SVM'].replace({1:1, 0:-1})
    df_all['randomforest'] = df_all['randomforest'].replace({1:1, 0:-1})

    # 将 StockGPT 的 0 值替换为 ML&DL 的预测结果
    for index, row in df_all.iterrows():
        StockGPT = row['StockGPT']
        if StockGPT == 0:
            col_mldl = [col for col in df_all.columns[3:] if col != 'StockGPT']
            df_mldl = row[col_mldl]
            pos_cnt = sum(df_mldl==1)
            neg_cnt = sum(df_mldl==-1)

            df_all.at[index, 'StockGPT'] = 1 if pos_cnt > len(col_mldl)//2 else -1

    print(f"df_all: {df_all}")
    df_all.to_excel(save_path, index=False)

if __name__ == '__main__':
    Fire(main)