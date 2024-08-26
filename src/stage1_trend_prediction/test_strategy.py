import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import pro
import json
import os
import matplotlib.pyplot as plt
from utils import *
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from fire import Fire
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


path = os.getcwd()
folder_path = path+'/db_file/'
file_path = f'sqlite:////{folder_path}'


def get_指标(rr, port):
    from itertools import accumulate
    def BARSLAST(series_1):
        return pd.Series(np.array(list(accumulate(~series_1, lambda x, y: (x + y) * y))), index=series_1.index)

    净值曲线 = 1 + rr.cumsum()
    时间跨度 = len(rr) / 12
    总收益 = rr.sum()
    年化收益 = 总收益 / 时间跨度
    年化波动 = rr.std() * (12 ** 0.5)
    夏普比率 = 年化收益 / 年化波动
    最大回撤 = abs((rr.cumsum() - rr.cumsum().cummax()).min())
    卡玛比率 = 年化收益 / 最大回撤
    最大下潜期 = BARSLAST(净值曲线 / 净值曲线.cummax() == 1).max()
    dd_index = pro.index_daily(ts_code='399300.SZ')  # 沪深300
    dd_index.index = pd.to_datetime(dd_index['trade_date'])
    dd_index = dd_index.sort_index()
    dd_index = dd_index['close']
    dd_index = dd_index.resample('M').last().pct_change()
    dd_index = dd_index[port.index[0]:port.index[-1]]
    基准总收益 = dd_index.sum()
    基准年化收益 = 基准总收益 / 时间跨度
    年化超额收益 = 年化收益 - 基准年化收益

    xx1 = ['时间跨度', '年化收益', '年化超额收益', '年化波动', '夏普比率', '最大回撤', '卡玛比率', '最大下潜期']
    xx2 = [时间跨度, 年化收益, 年化超额收益, 年化波动, 夏普比率, 最大回撤, 卡玛比率, 最大下潜期, 基准年化收益]
    return dict(zip(xx1, xx2))


def 获取指数(df_ports):
    df_上证指数 = pro.index_daily(ts_code='000001.SH')
    df_沪深300 = pro.index_daily(ts_code='399300.SZ')
    df_上证50 = pro.index_daily(ts_code='000016.SH')
    df_创业 = pro.index_daily(ts_code='399006.SZ')

    df_上证指数.index = pd.to_datetime(df_上证指数['trade_date'])
    df_沪深300.index = pd.to_datetime(df_沪深300['trade_date'])
    df_上证50.index = pd.to_datetime(df_上证50['trade_date'])
    df_创业.index = pd.to_datetime(df_创业['trade_date'])

    df_上证指数 = df_上证指数.sort_index()
    df_沪深300 = df_沪深300.sort_index()
    df_上证50 = df_上证50.sort_index()
    df_创业 = df_创业.sort_index()

    df_上证指数 = df_上证指数['close']
    df_沪深300 = df_沪深300['close']
    df_上证50 = df_上证50['close']
    df_创业 = df_创业['close']

    df_指数 = pd.concat([df_上证指数, df_沪深300, df_上证50, df_创业], axis=1)
    df_指数.columns = ['SCI', 'CSI300', 'SSE50', 'CNX']
    df_指数 = df_指数.resample('M').last().pct_change()
    df_指数 = df_指数[df_ports.index[0]:df_ports.index[-1]]

    return df_指数

def calculate_accuracy(df):
    accuracy_dict = {}

    for column in df.columns:
        if column in df.columns:
            accuracy = (df[column] == df['ground_truth']).mean()
            accuracy_dict[column] = accuracy
    return accuracy_dict

def main(tushare_token, stockgpt_mldl_path, save_dir, file_name):
    global pro
    token = tushare_token
    pro = ts.pro_api(token)

    多空类型 = '多空都可'  # 仅限做多,仅限做空 多空都可
    weight = '市值加权'  #市值加权,平均加权

    dd1 = pd.read_excel(stockgpt_mldl_path, engine='openpyxl')

    dd1['date'] = pd.to_datetime(dd1['date'])
    dd = dd1
    dd['stock_name'] = dd['stock_name'].replace('云海金属', '宝武镁业')

    dd = dd.rename(columns={'Transformers':'Bert','chatglm2_6b_greedy':'chatglm2','fingpt_greedy':'FinGPT','finma_greedy':'FinMA'})

    print(f"dd1: {dd1}")

    accuracy = {}

    if 多空类型 == '多空都可':
        for column in dd.columns[4:]:
            filtered_df = dd[dd[column] != 0]
            correct_predictions = (filtered_df['ground_truth'] == filtered_df[column]).sum() 
            total_predictions = len(filtered_df)  
            accuracy[column] = f'{round((float(correct_predictions) / total_predictions)*100,2)}%'
        
    elif 多空类型 == '仅限做多':
        for column in dd.columns[4:]:
            filtered_df = dd[(dd[column] == 1)]  
            correct_predictions = (filtered_df['ground_truth'] == filtered_df[column]).sum()
            print("correct_predictions",correct_predictions)
            total_predictions = len(dd[dd[column] == 1])
            print("total_predictions",total_predictions)
            accuracy[column] = f'{round((float(correct_predictions) / total_predictions)*100,2)}%'

    else :
        for column in dd.columns[4:]:
            filtered_df = dd[(dd[column] == -1)] 
            correct_predictions = (filtered_df['ground_truth'] == filtered_df[column]).sum()
            print("correct_predictions",correct_predictions)
            total_predictions = len(dd[dd[column] == -1])
            print("total_predictions",total_predictions)
            accuracy[column] = f'{round((float(correct_predictions) / total_predictions)*100,2)}%'


    accuracy_df = pd.DataFrame.from_dict(accuracy, orient='index', columns=['accuracy'])
    print(accuracy_df)


    field_names = dd.columns[4:].tolist()


    dd_stock = pro.stock_basic(exchange='')

    # temporary patch: changes to some stock info
    # TODO: remove tushare api dependencies
    change_stock_name = {
        "*ST东园": "东方园林",
        "中交设计": "祁连山",
        "广东建工": "粤水电",
        "金牌家居": "金牌厨柜",
        "*ST金科": "金科股份"
    }
    for k,v in change_stock_name.items():
        dd_stock["name"] = dd_stock["name"].replace([k], v)
    new_row = pd.Series(["000961.SZ", "000961", "中南建设", None, None, None, None, None, None, None], index=dd_stock.columns)
    dd_stock = dd_stock.append(new_row.to_frame().T)

    name_code_dict = dict(zip(dd_stock['name'], dd_stock['ts_code']))
    code_name_dict = dict(zip(dd_stock['ts_code'], dd_stock['name']))
    codes = dd_stock[dd_stock['name'].isin(dd['stock_name'])]['ts_code'].tolist()
    dd['stock_code'] = dd['stock_name'].apply(lambda x: name_code_dict.get(x))
    dd['next_month'] = pd.to_datetime(dd['date']) + pd.offsets.DateOffset(months=1) + pd.offsets.MonthEnd(1)

    print(f"\n[NOTICE] Loading db files, please wait a moment...\n")
    df_adj = get_data_by_sql(file_path, 'daily_adj', 'daily_adj', codes, '*')
    df_kline = get_data_by_sql(file_path, 'daily_kline', 'daily_kline', codes, '*')
    df_dailybasic = get_data_by_sql(file_path, 'dailybasic', 'dailybasic', codes,
                                    'ts_code,trade_date,total_mv,pe_ttm,pb,dv_ttm')

    df_adj = get_pivot_data(df_adj, 'adj_factor')
    df_close = get_pivot_data(df_kline, 'close')
    df_close = (df_close * df_adj / df_adj.loc[df_adj.index[-1]]).round(2)
    df_close.columns = [code_name_dict.get(x) for x in df_close.columns]
    MV = get_pivot_data(df_dailybasic, 'total_mv')
    MV.columns = [code_name_dict.get(x) for x in MV.columns]

    ports = []
    for field_name in tqdm(field_names):
        ddx = dd[['stock_name', 'next_month', field_name]].drop_duplicates(subset=['stock_name', 'next_month'],
                                                                        keep='first')  # 取第一个
        ddx = ddx.pivot(index='next_month', columns='stock_name', values=field_name).fillna(0)
        MV = MV.resample('M').last()
        MV = MV[ddx.columns]
        MV = MV[MV.index.isin(ddx.index)]

        dd_ret = df_close.resample('M').last().pct_change()
        dd_ret = dd_ret[ddx.columns]
        dd_ret = dd_ret[dd_ret.index.isin(ddx.index)]

        MV = MV.astype(float)
        ddx = ddx.astype(float)
        dd_ret = dd_ret.astype(float)

        if 多空类型 == '仅限做多':
            ddx = ddx * (ddx > 0)
        elif 多空类型 == '仅限做空':
            ddx = ddx * (ddx < 0)

        if weight =='平均加权':
            port = (dd_ret * ddx).sum(1) / ddx.abs().sum(1)  # 简单平均的组合收益率
    
        else:
            port = (dd_ret*ddx*(MV*ddx.abs()).div((MV*ddx.abs()).sum(1),axis=0)).sum(1) 
        ports.append(port)

    df_ports = pd.concat(ports, axis=1)
    df_ports.columns = field_names


    df_指数 = 获取指数(df_ports)


    指标s = []
    for field_name in field_names:
        指标s.append(get_指标(df_ports[field_name], port))
    for x in df_指数.columns:
        指标s.append(get_指标(df_指数[x], port))



    df_指标 = pd.DataFrame(指标s).T
    df_指标.columns = field_names + list(df_指数.columns)
    df_指标 = df_指标.round(3)
    print("df_指标",df_指标)

    d0 = pd.DataFrame(dict(zip(field_names, [[0]] * len(field_names))),
                    index=[df_ports.index[0] - pd.offsets.DateOffset(months=1) + pd.offsets.MonthEnd(0)])
    df_ports = pd.concat([d0, df_ports])

    d0 = pd.DataFrame(dict(zip(df_指数.columns, [[0]] * len(df_指数.columns))),
                    index=[df_指数.index[0] - pd.offsets.DateOffset(months=1) + pd.offsets.MonthEnd(0)])
    df_指数 = pd.concat([d0, df_指数])



    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = get_cmap('tab10')
    line_styles = ['-', '--', '-.', ':']


    lines_ports = []
    for i, col in enumerate(df_ports.columns):
        line, = ax.plot(df_ports[col].cumsum(), color=cmap(i % cmap.N), linestyle=line_styles[i % len(line_styles)])
        lines_ports.append(line)


    lines_指数 = []
    for i, col in enumerate(df_指数.columns):
        line, = ax.plot(df_指数[col].cumsum(), color=cmap((i+len(df_ports.columns)) % cmap.N),
                linestyle=line_styles[(i+len(df_ports.columns)) % len(line_styles)])
        lines_指数.append(line)
        

    lines = lines_ports + lines_指数

    labels = list(df_ports.columns) + list(df_指数.columns)
    sorted_lines_labels = sorted(zip(lines, labels), key=lambda x: x[0].get_ydata()[-1], reverse=True)
    sorted_lines, sorted_labels = zip(*sorted_lines_labels)


    for line, label in zip(lines, labels):
        line.set_linewidth(1)  
        if label == "Stock-Chain":
            line.set_linestyle('-')
            line.set_linewidth(2)  
            line.set_color('IndianRed')

    ax.legend(sorted_lines, sorted_labels, bbox_to_anchor=(1.12, 0.9), loc='upper right')

    ax.grid()
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

    ax.set_xlabel('Time',fontsize=13,fontweight='bold')
    ax.set_ylabel('Accumulated returns',fontsize=12,fontweight='bold')
    plt.show()

    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}/{file_name}_{weight}.png", dpi=500)
    df_指标.to_csv(f"{save_dir}/{file_name}_{weight}.csv")

    json_result = df_指标['StockGPT'].to_json()
    with open(os.path.join(save_dir, 'all_result.jsonl'), 'a+') as f:
        f.write(json.dumps(json_result) + '\n')

    print('-----finish-----\n\n')

if __name__ == '__main__':
    Fire(main)