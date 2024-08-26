import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime
import statsmodels.api as sm
from tqdm import tqdm
from loguru import logger
from sqlalchemy import create_engine
import sqlalchemy
import os
from itertools import accumulate
import time

pro = None


def get_ret_stats(df):
    pp = df['my_pos'] * (df['close'].pct_change().shift(-1))
    年化收益 = ((1 + pp).cumprod())[-2] ** (1 / (len(pp) / 252)) - 1
    年化波动率 = pp.std() * (252 ** 0.5)
    年化夏普率 = 年化收益 / 年化波动率
    最大回撤 = (1 - (((1 + pp).cumprod()) / ((1 + pp).cumprod().cummax()))).max()
    df['future_N_ret'] = (df['close'].shift(-20) - df['close']) / df['close']
    dd = df[(df['my_pos'].shift(1) == 0) & (df['my_pos'] == 1)]
    开仓后N天正收益概率 = (dd['future_N_ret'] > 0).sum() / len(dd)
    return 年化收益, 年化波动率, 年化夏普率, 最大回撤,开仓后N天正收益概率


def get_close_data(file_path, tp, code, start_date, end_date):
    if tp == 'stock':
        codes = ['随便搞一个元素', code]  # 可填入多个股票方便后续快速测试
        df_adj = get_data_by_sql(file_path, 'daily_adj', 'daily_adj', codes, '*')
        df_kline = get_data_by_sql(file_path, 'daily_kline', 'daily_kline', codes, '*')

        df_adj = get_pivot_data(df_adj, 'adj_factor')
        df_close = get_pivot_data(df_kline, 'close')
        df_close = (df_close * df_adj / df_adj.loc[df_adj.index[-1]]).round(2)

        df_high = get_pivot_data(df_kline, 'high')
        df_high = (df_high * df_adj / df_adj.loc[df_adj.index[-1]]).round(2)

        df_low = get_pivot_data(df_kline, 'low')
        df_low = (df_low * df_adj / df_adj.loc[df_adj.index[-1]]).round(2)

        df_dailybasic = get_data_by_sql(file_path, 'dailybasic', 'dailybasic', codes, 'trade_date,ts_code,pe_ttm')
        df_dailybasic = df_dailybasic.copy().sort_values(by=['ts_code', 'trade_date']).reset_index(drop=True)
        df_dailybasic = df_dailybasic.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last').reset_index(
            drop=True)
        df_dailybasic = get_pivot_data(df_dailybasic, 'pe_ttm')

        df = pd.concat([df_close[code], df_high[code], df_low[code], df_dailybasic[code]], axis=1)
        df.columns = ['close', 'high', 'low', 'pe_ttm']
        df = df.sort_index()
        df = df.fillna(method='ffill')

    elif tp == 'sw':
        df = pro.sw_daily(ts_code=code, start_date=start_date, end_date=end_date)
        df.index = pd.to_datetime(df['trade_date'])
        df = df.sort_index()

    elif tp == 'index':
        df = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
        df.index = pd.to_datetime(df['trade_date'])
        df = df.sort_index()
    else:
        print('tp参数错误')
    return df


def func_KDJ(df):
    low_list = df['low'].rolling(9, min_periods=1).min()
    high_list = df['high'].rolling(9, min_periods=1).max()
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    K = df['K']
    D = df['D']
    J = df['J']
    KDJ金叉 = (J > K) & (J > D) & (J.shift(1) < K.shift(1)) & (J.shift(1) < D.shift(1))
    KDJ死叉 = (J < K) & (J < D) & (J.shift(1) > K.shift(1)) & (J.shift(1) > D.shift(1))
    df['KDJ金叉'] = KDJ金叉
    df['KDJ死叉'] = KDJ死叉
    return df


def func_MACD(df):
    df['DIF'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['DEA'] = df['DIF'].ewm(span=9).mean()
    df['MACD'] = df['DIF'] - df['DEA']
    return df.copy()


def get_calendar(freq, start_date, end_date):
    calendar = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
    calendar = calendar[calendar['is_open'] == 1]
    calendar = calendar.sort_values(by=['cal_date']).reset_index(drop=True)
    calendar['date'] = pd.to_datetime(calendar['cal_date'])
    calendar['week'] = calendar['date'].apply(lambda x: '-'.join([str(x1).zfill(2) for x1 in x.isocalendar()[:2]]))
    calendar['month'] = calendar['cal_date'].apply(lambda x: x[:4] + '-' + x[4:6])

    if freq == 'weekly':
        calendar = calendar.drop_duplicates(subset=['week'], keep='last').reset_index(drop=True)
    elif freq == 'monthly':
        calendar = calendar.drop_duplicates(subset=['month'], keep='last').reset_index(drop=True)
    return calendar


def BARSLAST(series_1):
    """
    处理一个bool序列
    计算当前与上一次为True的距离
    """
    return pd.Series(list(accumulate(~series_1, lambda x, y: (x + y) * y)), index=series_1.index)


def get_stocks():
    dd1 = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd2 = pro.stock_basic(list_status='D', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd3 = pro.stock_basic(list_status='P', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd1['status'] = '上市'
    dd2['status'] = '退市'
    dd3['status'] = '暂停'
    dd_stocks = pd.concat([dd1, dd2, dd3])
    # 去掉一些奇怪的代码
    dd_stocks = dd_stocks[~dd_stocks['ts_code'].isin(['T00018.SH'])]

    dd_stocks = dd_stocks.reset_index(drop=True)
    dd_stocks['delist_date'] = dd_stocks['delist_date'].fillna('20991231')
    return dd_stocks


def get_tradeable_stocks(start_date, end_date, indus_code, delete_ST):
    dd1 = pro.stock_basic(list_status='L', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd2 = pro.stock_basic(list_status='D', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd3 = pro.stock_basic(list_status='P', fields='ts_code,symbol,name,area,industry,list_date,delist_date')
    dd1['status'] = '上市'
    dd2['status'] = '退市'
    dd3['status'] = '暂停'
    dd_stocks = pd.concat([dd1, dd2, dd3])
    # 去掉一些奇怪的代码
    dd_stocks = dd_stocks[~dd_stocks['ts_code'].isin(['T00018.SH'])]

    dd_stocks = dd_stocks.reset_index(drop=True)
    dd_stocks['delist_date'] = dd_stocks['delist_date'].fillna('20991231')

    上市时间df = dd_stocks.copy()

    dd_nameschange = get_namechange('data/nameschange.csv', 上市时间df)
    dd_停复牌 = get_停复牌('data/停复牌.csv')
    dd_停复牌 = dd_停复牌[dd_停复牌['suspend_type'] == 'S']
    dd_swmember = get_sw_members('data/申万行业分类与成分.csv')

    df_calendar = pro.trade_cal(exchange='')
    df_calendar = df_calendar[df_calendar['is_open'] == 1]
    df_calendar = df_calendar[(df_calendar['cal_date'] >= start_date) & (df_calendar['cal_date'] < end_date)]
    df_calendar = df_calendar.reset_index(drop=True)
    df_calendar.index = pd.to_datetime(df_calendar['cal_date'])

    print('遍历所有股票过滤ST数据')
    dds = []
    for i in tqdm(range(len(上市时间df))):
        code = 上市时间df.loc[i, 'ts_code']
        dds.append(get_ST_dates(df_calendar, dd_nameschange, code))

    df_ST = pd.concat(dds, axis=1).fillna(0)
    df_ST.columns = 上市时间df['ts_code']
    df_ST = df_ST.sort_index()
    df_ST = 1 - df_ST

    print('遍历所有交易日')
    pp = []
    for i in tqdm(range(len(df_calendar))):
        the_date = df_calendar.loc[df_calendar.index[i], 'cal_date']
        codes = get_codes_by_date(上市时间df, dd_停复牌, dd_swmember, df_ST,
                                  the_date, indus_code=indus_code, delete_ST=delete_ST)
        pp.append(dict(zip(codes, [1] * len(codes))))

    df = pd.DataFrame(pp, index=df_calendar['cal_date'])
    df = df[sorted(df.columns)]
    df = df.fillna(0)
    df = df.astype(int)
    return df


def get_codes_by_date(上市时间df, dd_停复牌, dd_swmember, df_ST,
                      the_date, indus_code=None, delete_ST=False):
    """
    给定交易日,根据上市退市时间，停复牌时间，ST股票时间,行业代码成分股时间，给出可选的股票篮子
    the_date:'20201008'
    indus_code = '850531.SI' 或 None
    delete_ST = True 或 False
    """
    在上市股票 = list(上市时间df[(上市时间df['list_date'] <= the_date) & (上市时间df['delist_date'] >= the_date)]['ts_code'])
    # 去除北交所股票
    在上市股票 = list(filter(lambda x: not x.endswith('.BJ'), 在上市股票))
    # 去除停牌股票
    停牌股票 = list(dd_停复牌[dd_停复牌['trade_date'] == the_date]['ts_code'])
    可选股票 = list(set(在上市股票) - set(停牌股票))

    if indus_code and delete_ST:
        non_ST_codes = df_ST.loc[pd.to_datetime(the_date)]
        非ST股票 = list(non_ST_codes[non_ST_codes == 1].keys())
        行业股票 = list(dd_swmember[(dd_swmember['index_code'] == indus_code) &
                                (dd_swmember['in_date'] <= the_date) &
                                (dd_swmember['out_date'] > the_date)]['con_code'])
        可选股票 = list(set(可选股票) & set(行业股票) & set(非ST股票))

    elif indus_code:
        行业股票 = list(dd_swmember[(dd_swmember['index_code'] == indus_code) &
                                (dd_swmember['in_date'] <= the_date) &
                                (dd_swmember['out_date'] > the_date)]['con_code'])
        可选股票 = list(set(可选股票) & set(行业股票))

    elif delete_ST:
        non_ST_codes = df_ST.loc[pd.to_datetime(the_date)]
        非ST股票 = list(non_ST_codes[non_ST_codes == 1].keys())
        可选股票 = list(set(可选股票) & set(非ST股票))
    return 可选股票


def get_ST_dates(df_calendar, dd_nameschange, code):
    dd = dd_nameschange[dd_nameschange['ts_code'] == code]
    dd = dd.fillna('缺失值')
    dd['date'] = \
        (dd['ann_date'] != '缺失值') * (dd['ann_date'].astype(str)) + \
        (dd['ann_date'] == '缺失值') * (dd['start_date'].astype(str))
    dd['date'] = dd['date'].astype(float).astype(int).astype(str)
    dd = dd.sort_values(by=['date']).reset_index(drop=True)
    dd['ST'] = dd['name'].apply(lambda x: 'ST' in x)
    dd = dd[dd['ST'] != dd['ST'].shift(1)].reset_index(drop=True)

    pp = []
    p = []
    for i in range(len(dd)):
        if dd.loc[i, 'ST']:
            p = []
            p.append(dd.loc[i, 'date'])

        if len(p) > 0 and not dd.loc[i, 'ST']:
            p.append(dd.loc[i, 'date'])
            pp.append(p)
            p = []

        if len(p) == 2 or i == len(dd) - 1:
            pp.append(p)

    pp = list(filter(lambda x: len(x) > 0, pp))

    ST_dates = []
    for p in pp:
        if len(p) == 2:
            dates = list(df_calendar[(df_calendar['cal_date'] >= p[0]) & (df_calendar['cal_date'] <= p[1])]['cal_date'])
        elif len(p) == 1:
            dates = list(df_calendar[(df_calendar['cal_date'] >= p[0])]['cal_date'])
        ST_dates += dates

    return df_calendar[df_calendar.index.isin(ST_dates)]['is_open']


def get_namechange(file_path, dd_stocks):
    if not os.path.exists(file_path):
        logger.info('在所有股票中循环，获取每个时间下的改名数据')
        # 重新获取改名数据
        dds = []
        for ts_code in tqdm(dd_stocks['ts_code']):
            dd = pro.namechange(ts_code=ts_code)
            dds.append(dd)
        dd_nameschange = pd.concat(dds)
        dd_nameschange.to_csv(file_path, index=False)

    # 直接获取保存的改名数据
    dd_nameschange = pd.read_csv(file_path)
    dd_nameschange = dd_nameschange[dd_nameschange['change_reason'].isin(
        ['*ST', 'ST', '终止上市', '撤销ST', '撤销*ST'])]
    dd_nameschange = dd_nameschange.fillna('缺失值')
    dd_nameschange['date'] = \
        (dd_nameschange['ann_date'] != '缺失值') * (dd_nameschange['ann_date'].astype(str)) + \
        (dd_nameschange['ann_date'] == '缺失值') * (dd_nameschange['start_date'].astype(str))
    dd_nameschange['date'] = dd_nameschange['date'].astype(float).astype(int).astype(str)

    dd_nameschange = dd_nameschange.sort_values(by=['ts_code', 'date']).reset_index(drop=True)
    dd_nameschange['ST'] = dd_nameschange['change_reason'].apply(lambda x: '撤销' not in x)
    dd_nameschange = dd_nameschange.drop_duplicates(subset=['ts_code', 'date']).reset_index(drop=True)
    return dd_nameschange


def get_停复牌(file_path):
    if not os.path.exists(file_path):
        logger.info('在给定年份范围中循环，获取每个时间下的停复牌数据')
        range_years = list(range(2000, 2022))
        dd1s = []
        dd2s = []
        dd3s = []
        dd4s = []
        for year in tqdm(range_years):
            for month in range(1, 13):
                year = str(year)
                month = str(month).zfill(2)
                year_month = year + month

                start_date = year_month + '01'
                end_date = year_month + '15'
                dd1 = pro.suspend_d(suspend_type='S', start_date=start_date, end_date=end_date)
                dd2 = pro.suspend_d(suspend_type='R', start_date=start_date, end_date=end_date)
                dd1s.append(dd1)
                dd2s.append(dd2)

                start_date = year_month + '16'
                end_date = year_month + '31'
                dd3 = pro.suspend_d(suspend_type='S', start_date=start_date, end_date=end_date)
                dd4 = pro.suspend_d(suspend_type='R', start_date=start_date, end_date=end_date)
                dd3s.append(dd3)
                dd4s.append(dd4)

        dd = pd.concat(dd1s + dd2s + dd3s + dd4s)
        dd = dd.drop(['suspend_timing'], axis=1)
        停复牌df = dd.copy()
        停复牌df.to_csv(file_path, index=False)

    停复牌df = pd.read_csv(file_path)
    停复牌df = 停复牌df.sort_values(by=['ts_code', 'trade_date'])
    停复牌df = 停复牌df[停复牌df['suspend_type'] == 'S']
    停复牌df = 停复牌df.reset_index(drop=True)
    停复牌df = 停复牌df.astype(str)
    return 停复牌df


def get_sw_members(file_path):
    if not os.path.exists(file_path):
        logger.info('在所有申万行业一二三级子分类中循环，获取每个分类下的申万成分数据')
        dd1 = pro.index_classify(level='L1', src='SW2021')
        dd2 = pro.index_classify(level='L2', src='SW2021')
        dd3 = pro.index_classify(level='L3', src='SW2021')
        dd_sw = pd.concat([dd1, dd2, dd3])
        dd_sw = dd_sw.reset_index(drop=True)

        dds = []
        for i in tqdm(range(len(dd_sw))):
            index_code = dd_sw.loc[i, 'index_code']
            industry_name = dd_sw.loc[i, 'industry_name']
            level = dd_sw.loc[i, 'level']
            dd1 = pro.index_member(index_code=index_code, is_new='Y')
            dd2 = pro.index_member(index_code=index_code, is_new='N')
            dd = pd.concat([dd1, dd2])
            dd = dd.reset_index(drop=True)
            dd['out_date'] = dd['out_date'].fillna('20991231')
            dd['index_code'] = index_code
            dd['industry_name'] = industry_name
            dd['level'] = level
            dds.append(dd)
        df = pd.concat(dds)
        df.to_csv(file_path, index=False)
    df = pd.read_csv(file_path)
    df[['in_date', 'out_date']] = df[['in_date', 'out_date']].astype(str)
    return df


def get_codes(index_code='000300.SH'):
    """
    获取最新的指数成分股
    """
    df_stocks = pro.index_weight(index_code=index_code)
    df_stocks = df_stocks[df_stocks['trade_date'] == df_stocks.loc[0, 'trade_date']]
    df_stocks = df_stocks.sort_values(by=['con_code']).reset_index(drop=True)
    codes = list(df_stocks['con_code'])
    return codes


def get_index_close(index_code='000300.SZ', start_date='20100101'):
    """
    获取指数收盘价
    """
    dd_index = pro.index_daily(ts_code=index_code, start_date=start_date)
    dd_index.index = pd.to_datetime(dd_index['trade_date'])
    dd_index = dd_index.sort_index()
    return dd_index


def get_data_by_sql(file_path, db_name, table_name, codes, fields):
    """
    输入表名和股票篮子数组，
    输出从数据库中读取的数据
    """
    codes = codes + ['随便写什么']
    database_name = f'{db_name}.db'
    engine = create_engine(file_path + database_name)
    sql = f"select {fields} from {table_name} where ts_code in {tuple(codes)}"
    dd = pd.read_sql(sql, engine)
    # 如果是财报数据，将实际公布时间设为trade_date字段
    if 'f_ann_date' in dd.columns or 'ann_date' in dd.columns:
        dd['trade_date'] = dd['end_date']
    dd['trade_date'] = pd.to_datetime(dd['trade_date'])
    dd = dd.sort_index()
    return dd


def get_pivot_data(dd, field):
    """
    指定一个字段名，然后将数据变成index是日期，columns是该字段下的变量
    输入df和需要按日期分行列名，
    输出pivot后的df
    """
    dd = dd.pivot(index='trade_date', columns='ts_code', values=[field])
    cols = list(map(lambda x: x[1], dd.columns))
    dd.columns = cols
    return dd


def halflife_weighting(x, halflife):
    """
    输入df和指定半衰期，对一个df或series计算指数加权均值
    """
    # 该公式，参考网址：
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ewm.html
    # 基于里面说的alpha,span,halflife的关系得出下面公式
    span = 2 / (1 - np.exp(np.log(0.5) / halflife)) - 1
    return x.ewm(span=span).mean()


def get_adj_data(code, freq='daily', start_date=None, end_date=None):
    if freq == 'daily':
        df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    elif freq == 'monthly':
        df = pro.monthly(ts_code=code, start_date=start_date, end_date=end_date)

    if len(df) > 1:
        df = df.sort_values(by=['trade_date']).reset_index(drop=True)
        df = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
        df_factor = pro.adj_factor(ts_code=code)
        df = pd.merge(df, df_factor, on=['trade_date'])
        now_adj = df.loc[len(df) - 1, 'adj_factor']
        df['adjs'] = df['adj_factor'] / now_adj
        df_adjs = pd.concat([df['adjs']] * 4, axis=1)
        df_adjs.columns = ['open', 'high', 'low', 'close']
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] * df_adjs
    else:
        pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'amount'])
    return df


def update_dailybasic_data(table_name, cal_dates, engine):
    # 获取数据库中最新的数据日期
    try:
        exist_dates = pd.read_sql(f"select distinct trade_date from {table_name}", engine)
        cal_dates = sorted(list(set(cal_dates) - set(exist_dates['trade_date'])))
    except:
        pass

    logger.info(f'需要更新{len(cal_dates)}个交易日的数据')
    for cal_date in tqdm(cal_dates):
        df = pro.daily_basic(trade_date=cal_date)
        df.to_sql(table_name, engine, index=True, if_exists='append')


def update_dailykline_data(table_name, cal_dates, engine):
    # 获取数据库中最新的数据日期
    try:
        exist_dates = pd.read_sql(f"select distinct trade_date from {table_name}", engine)
        cal_dates = sorted(list(set(cal_dates) - set(exist_dates['trade_date'])))
    except:
        pass

    logger.info(f'需要更新{len(cal_dates)}个交易日的数据')
    for cal_date in tqdm(cal_dates):
        df = pro.daily(trade_date=cal_date)
        df.to_sql(table_name, engine, index=True, if_exists='append')


def update_dailyadj_data(table_name, cal_dates, engine):
    # 获取数据库中最新的数据日期
    try:
        exist_dates = pd.read_sql(f"select distinct trade_date from {table_name}", engine)
        cal_dates = sorted(list(set(cal_dates) - set(exist_dates['trade_date'])))
    except:
        pass

    logger.info(f'需要更新{len(cal_dates)}个交易日的数据')
    for cal_date in tqdm(cal_dates):
        df = pro.adj_factor(trade_date=cal_date)
        df.to_sql(table_name, engine, index=True, if_exists='append')


def update_financial_data(table_name, end_dates, engine):
    try:
        # 如果table中已经有了，就不必再重复获取
        # 且获取已经存在的table中的所有字段fields，确保新增的数据不会多出奇怪的字段
        exist_dates = pd.read_sql(f"select distinct end_date from {table_name}", engine)
        end_dates = sorted(list(set(end_dates) - set(exist_dates['end_date'])))

        md = sqlalchemy.MetaData()
        table = sqlalchemy.Table(table_name, md, autoload=True, autoload_with=engine)
        fields = [c.name for c in table.c]
        fields = list(filter(lambda x: x != 'level_0', fields))
        logger.info(f'fields:{fields}')

    except:
        pass

    logger.info(f'需要更新{len(end_dates)}个财报日的数据')
    for end_date in tqdm(end_dates):
        if table_name == '利润表':
            dd1 = pro.income_vip(period=end_date, report_type='1')
            # print('dd1',dd1)
            dd2 = pro.income_vip(period=end_date, report_type='2')
            # print('dd2',dd2)
            dd = pd.concat([dd1, dd2])
        elif table_name == '现金流量表':
            dd1 = pro.cashflow_vip(period=end_date, report_type='1')
            dd2 = pro.cashflow_vip(period=end_date, report_type='2')
            dd = pd.concat([dd1, dd2])
        elif table_name == '资产负债表':
            dd = pro.balancesheet_vip(period=end_date)

        dd = dd.reset_index()

        # 筛选已有的字段
        # dd = dd[fields]

        dd.to_sql(table_name, engine, index=True, if_exists='append')


def combine_3_fin_data(利润表, 现金流量表, 资产负债表):
    利润表['code_date'] = 利润表['ts_code'] + '_' + 利润表['end_date']
    利润表 = 利润表.sort_values(['code_date', 'f_ann_date']).reset_index(drop=True)
    利润表 = 利润表.drop_duplicates(subset=['code_date'], keep='last')

    现金流量表['code_date'] = 现金流量表['ts_code'] + '_' + 现金流量表['end_date']
    现金流量表 = 现金流量表.sort_values(['code_date', 'f_ann_date']).reset_index(drop=True)
    现金流量表 = 现金流量表.drop_duplicates(subset=['code_date'], keep='last')

    资产负债表['code_date'] = 资产负债表['ts_code'] + '_' + 资产负债表['end_date']
    资产负债表 = 资产负债表.sort_values(['code_date', 'f_ann_date']).reset_index(drop=True)
    资产负债表 = 资产负债表.drop_duplicates(subset=['code_date'], keep='last')

    利润表.index = 利润表['code_date']
    现金流量表.index = 现金流量表['code_date']
    资产负债表.index = 资产负债表['code_date']

    drop_cols = ['level_0', 'index', 'ts_code', 'ann_date', 'end_date', 'report_type', 'comp_type', 'update_flag',
                 'code_date']

    利润表 = 利润表.drop(drop_cols, axis=1)
    现金流量表 = 现金流量表.drop(drop_cols, axis=1)
    资产负债表 = 资产负债表.drop(drop_cols, axis=1)

    利润表.rename(columns={'f_ann_date': '利润表公布时间'}, inplace=True)
    现金流量表.rename(columns={'f_ann_date': '现金流量表公布时间'}, inplace=True)
    资产负债表.rename(columns={'f_ann_date': '资产负债表公布时间'}, inplace=True)

    dff = pd.concat([利润表, 现金流量表, 资产负债表], axis=1)
    dff = dff.reset_index()
    dff['ts_code'] = dff['code_date'].apply(lambda x: x.split('_')[0])
    dff['end_date'] = dff['code_date'].apply(lambda x: x.split('_')[1])
    dff = dff.drop(['code_date'], axis=1)
    dff = dff[['ts_code', 'end_date'] + list(dff.columns)[:-2]]

    return dff


def update_hkhold_data(table_name, cal_dates, engine):
    # 获取数据库中最新的数据日期
    try:
        exist_dates = pd.read_sql(f"select distinct trade_date from {table_name}", engine)
        cal_dates = sorted(list(set(cal_dates) - set(exist_dates['trade_date'])))
    except:
        pass

    logger.info(f'需要更新{len(cal_dates)}个交易日的数据')
    for cal_date in tqdm(cal_dates):
        df = pro.hk_hold(trade_date=cal_date)
        df.to_sql(table_name, engine, index=True, if_exists='append')
        time.sleep(0.15)  # 每分钟最多400次


def update_moneyflow_data(table_name, cal_dates, engine):
    # 获取数据库中最新的数据日期
    try:
        exist_dates = pd.read_sql(f"select distinct trade_date from {table_name}", engine)
        cal_dates = sorted(list(set(cal_dates) - set(exist_dates['trade_date'])))
    except:
        pass

    logger.info(f'需要更新{len(cal_dates)}个交易日的数据')
    for cal_date in tqdm(cal_dates):
        df = pro.moneyflow(trade_date=cal_date)
        df.to_sql(table_name, engine, index=True, if_exists='append')


def get_fin_data(end_date, report_type):
    dd1 = pro.income_vip(period=end_date, report_type=report_type)
    dd2 = pro.cashflow_vip(period=end_date, report_type=report_type)
    dd3 = pro.balancesheet_vip(period=end_date)

    if len(dd1) > 0:
        dd1 = dd1[dd1['ts_code'].apply(lambda x: x.split('.')[1] in ['SH', 'SZ'])]
        dd2 = dd2[dd2['ts_code'].apply(lambda x: x.split('.')[1] in ['SH', 'SZ'])]
        dd3 = dd3[dd3['ts_code'].apply(lambda x: x.split('.')[1] in ['SH', 'SZ'])]
        dd1 = dd1.sort_values(by=['f_ann_date']).reset_index(drop=True).drop_duplicates(subset=['ts_code', 'end_date'],
                                                                                        keep='last')
        dd2 = dd2.sort_values(by=['f_ann_date']).reset_index(drop=True).drop_duplicates(subset=['ts_code', 'end_date'],
                                                                                        keep='last')
        dd3 = dd3.sort_values(by=['f_ann_date']).reset_index(drop=True).drop_duplicates(subset=['ts_code', 'end_date'],
                                                                                        keep='last')
        dd1['code_date'] = dd1['ts_code'] + '_' + dd1['end_date']
        dd2['code_date'] = dd2['ts_code'] + '_' + dd2['end_date']
        dd3['code_date'] = dd3['ts_code'] + '_' + dd3['end_date']
        dd1.index = dd1['code_date']
        dd2.index = dd2['code_date']
        dd3.index = dd3['code_date']

        # 下面的删除字段有一点讲究，dd3是资产负债，report_type必然是‘1’，因此保留的report_type需要在dd1或dd2中
        dd1 = dd1.drop(['ts_code', 'end_date', 'comp_type', 'report_type', 'ann_date', 'f_ann_date', 'code_date'],
                       axis=1)
        dd2 = dd2.drop(['ts_code', 'end_date', 'comp_type', 'ann_date', 'f_ann_date', 'code_date'], axis=1)
        dd3 = dd3.drop(['comp_type', 'report_type', 'ann_date', 'code_date'], axis=1)
        dd = pd.concat([dd1, dd2, dd3], axis=1)

    else:
        dd = pd.concat([dd1, dd2, dd3], axis=1)
    return dd


def get_data(code):
    dd1 = pro.income(ts_code=code)
    dd2 = pro.balancesheet(ts_code=code)
    dd3 = pro.cashflow(ts_code=code)
    dd4 = pro.fina_indicator(ts_code=code)

    dd1 = dd1.drop(['ebit', 'ebitda'], axis=1)

    dd1 = dd1.sort_values(by=['ann_date']).reset_index(drop=True)
    dd2 = dd2.sort_values(by=['ann_date']).reset_index(drop=True)
    dd3 = dd3.sort_values(by=['ann_date']).reset_index(drop=True)
    dd4 = dd4.sort_values(by=['ann_date']).reset_index(drop=True)

    dd1 = dd1.drop_duplicates(subset=['f_ann_date'], keep='last').reset_index(drop=True)
    dd2 = dd2.drop_duplicates(subset=['f_ann_date'], keep='last').reset_index(drop=True)
    dd3 = dd3.drop_duplicates(subset=['f_ann_date'], keep='last').reset_index(drop=True)
    dd4 = dd4.drop_duplicates(subset=['ann_date'], keep='last').reset_index(drop=True)

    dd1 = dd1.drop(['ts_code', 'f_ann_date', 'end_date', 'report_type', 'comp_type'], axis=1)
    dd2 = dd2.drop(['ts_code', 'f_ann_date', 'end_date', 'report_type', 'comp_type'], axis=1)
    dd3 = dd3.drop(['ts_code', 'f_ann_date', 'end_date', 'report_type', 'comp_type'], axis=1)
    dd4 = dd4.drop(['ts_code', 'end_date'], axis=1)

    dd = pd.merge(dd1, dd2, on=['ann_date'], how='outer')
    dd = pd.merge(dd, dd3, on=['ann_date'], how='outer')
    dd = pd.merge(dd, dd4, on=['ann_date'], how='outer')

    dd.rename(columns={'ann_date': 'datetime'}, inplace=True)
    dd = dd.sort_values(['datetime']).reset_index(drop=True)

    # dd_dates = pro.trade_cal(exchange='SSE',start_date=dd.loc[0,'datetime'],end_date=dd.loc[dd.index[-1],'datetime'])
    dd_dates = pro.trade_cal(exchange='SSE', start_date=dd.loc[0, 'datetime'],
                             end_date=datetime.strftime(datetime.now(), '%Y%m%d'))
    dd_dates = dd_dates[['cal_date']]
    dd_dates.columns = ['datetime']

    dd = pd.merge(dd, dd_dates, on=['datetime'], how='right').fillna(method='ffill')

    dd5 = pro.daily_basic(ts_code=code)
    # 每日指标中的total_share应该是基于标准财报日的，而dd2中资产负债表中的total_share按照公告日期更新
    dd5 = dd5.drop(['ts_code', 'total_share'], axis=1)
    dd5.rename(columns={'trade_date': 'datetime'}, inplace=True)
    dd5 = dd5.sort_values(by=['datetime']).reset_index(drop=True).fillna(method='ffill')

    dd = pd.merge(dd, dd5, on=['datetime'], how='left').fillna(method='ffill')
    return dd


def get_holder_data(ts_code):
    # 输入股票代码，输出每一年的第1大股东持股比例与第2,3大股东持股比例之和的比值
    dd = pro.top10_holders(ts_code=ts_code)
    dd = dd.sort_values(by=['end_date', 'holder_name']).reset_index(drop=True)
    dd['year'] = dd['end_date'].apply(lambda x: x[:4])
    dd['holder_name_end_date'] = dd['holder_name'] + '_' + dd['end_date']
    dd = dd.sort_values(by=['end_date', 'holder_name']).reset_index(drop=True)
    look_dates = dd.drop_duplicates(subset=['year'], keep='last')['end_date'].unique()  # 取每一年的最后的end_date作为参考日
    dd = dd[dd['end_date'].isin(look_dates)]
    dd = dd.sort_values(by=['year', 'hold_ratio'], ascending=[True, False]).reset_index(drop=True)
    # 直接用.cumcount()而不是.apply(lambda x:x.rank(ascending=False))，是因为rank可能因持股比例相同而产生不是顺序的整数，如1,2,4,4
    dd['year_rank'] = dd[['year', 'hold_ratio']].groupby(['year']).cumcount() + 1
    dd.index = dd['year']
    dd1 = dd[dd['year_rank'] == 1]['hold_ratio']
    dd2 = dd[dd['year_rank'] == 2]['hold_ratio']
    dd3 = dd[dd['year_rank'] == 3]['hold_ratio']
    return dd1 / (dd2 + dd3)


def get_agency_data(ts_code):
    dd = pro.fina_audit(ts_code=ts_code).drop_duplicates().sort_values(by=['end_date']).reset_index(drop=True)
    dd = dd[dd['end_date'].apply(lambda x: x[-4:] == '1231')]
    if len(dd) > 0:
        dd['year'] = dd['end_date'].apply(lambda x: x[:4])
        dd.index = dd['year']
        dd['是否更换律所'] = (dd['audit_agency'] != dd['audit_agency'].shift(1)).apply(lambda x: 1 if x else 0)
        dd['是否未被出具非标意见'] = (dd['audit_result'] != '标准无保留意见').apply(lambda x: 1 if x else 0)
        return dd['是否更换律所'], dd['是否未被出具非标意见']
    else:
        return pd.Series([], name='是否更换律所', dtype=int), pd.Series([], name='是否未被出具非标意见', dtype=int)


def addfield(dd, code):
    补充字段 = ['dtprofit_to_profit', 'q_profit_to_gr', 'q_profit_yoy', 'q_gsprofit_margin']
    cols = ['ann_date'] + 补充字段
    dd4 = pro.fina_indicator(ts_code=code, fields=','.join(cols))
    dd4 = dd4.sort_values(by=['ann_date']).reset_index(drop=True)
    dd4 = dd4.drop_duplicates(subset=['ann_date'], keep='last').reset_index(drop=True)
    dd4.rename(columns={'ann_date': 'datetime'}, inplace=True)
    dd = pd.merge(dd, dd4, on=['datetime'], how='outer')
    dd = dd.sort_values(['datetime']).reset_index(drop=True)
    dd_dates = pro.trade_cal(exchange='SSE', start_date=dd.loc[0, 'datetime'],
                             end_date=datetime.strftime(datetime.now(), '%Y%m%d'))
    dd_dates = dd_dates[['cal_date']]
    dd_dates.columns = ['datetime']
    dd = pd.merge(dd, dd_dates, on=['datetime'], how='right').fillna(method='ffill')
    return dd


def SMA(series_1, N, M):
    '''

    :param series_1: 一个pd.Series
    :param N:窗口长度，求均值时被当作分母
    :param M:对新值分配的权重
    :return:一个数组
    '''
    pp_valid = []
    list_1 = list(series_1)
    pp_nan = []
    for i in range(len(list_1)):
        if np.isnan(list_1[i]):
            pp_nan.append(np.nan)
        else:
            pp_valid = list_1[i:]
            break

    if len(pp_valid) == 0:
        return [np.nan] * len(series_1)

    pp = [pp_valid[0]]
    for i in range(1, len(pp_valid)):
        past = pp[-1]
        pp.append((M * pp_valid[i] + (N - M) * past) / N)
    return pp_nan + pp


def factors1(code):
    dd_index = pro.index_monthly(ts_code='000001.SH')
    dd_index.rename(columns={'trade_date': 'datetime'}, inplace=True)
    dd_index = dd_index.sort_values(by=['datetime']).reset_index(drop=True)
    dd_code = get_adj_data(code=code, freq='monthly')
    dd_code.rename(columns={'trade_date': 'datetime'}, inplace=True)
    dd_code['datetime'] = dd_code['datetime'].astype(str)
    dd_index['datetime'] = dd_index['datetime'].astype(str)
    dd = pd.merge(dd_code[['datetime', 'close']], dd_index[['datetime', 'close']], on=['datetime'], how='right').fillna(
        method='ffill').dropna()
    dd.columns = ['datetime', 'code', 'index']
    dd = dd.reset_index(drop=True)
    dd['code_ret'] = dd['code'].pct_change()
    dd['index_ret'] = dd['index'].pct_change()

    if len(dd) > 60:
        p_alpha = [np.nan] * 60
        p_beta = [np.nan] * 60
        for ii in range(60, len(dd)):
            x = dd.loc[ii - 60 + 1:ii, 'code_ret']
            y = dd.loc[ii - 60 + 1:ii, 'index_ret']

            x = sm.add_constant(x)
            model = sm.OLS(y, x)
            result = model.fit()
            p_alpha.append(result.params['const'])
            p_beta.append(result.params['code_ret'])

        dd['HAlpha'] = p_alpha
        dd['beta'] = p_beta
    else:
        dd['HAlpha'] = np.nan
        dd['beta'] = np.nan

    for N in [1, 3, 6, 12]:
        dd[f'return_{N}m'] = (dd['code'] - dd['code'].shift(N)) / (dd['code'].shift(N))

    dd['month'] = dd['datetime'].apply(lambda x: x[:6])
    return dd


def factors2(folder, code):
    file = f'{code}.csv'
    dd = pd.read_csv(f'{folder}/{file}')
    dd['datetime'] = dd['datetime'].astype(str)

    dd0 = get_adj_data(code, freq='daily')
    dd0 = dd0[['trade_date', 'close']]
    dd0.columns = ['datetime', 'adj_close']
    dd = pd.merge(dd, dd0, on=['datetime'], how='right')[['datetime', 'adj_close', 'turnover_rate', 'total_mv']]
    dd = dd.dropna()

    exp_func = lambda x: (x.values * np.exp(-np.arange(N * 21)[::-1] / 12 / 4)).sum()
    for N in [1, 3, 6, 12]:
        dd[f'turn_{N}m'] = dd['turnover_rate'].rolling(21 * N).mean()
        dd[f'bias_turn_{N}m'] = dd['turnover_rate'].rolling(21 * N).mean() / dd['turnover_rate'].rolling(
            21 * 12 * 2).mean()
        dd[f'wgt_return_{N}m'] = (dd['adj_close'].pct_change() * dd['turnover_rate']).rolling(21 * N).mean()
        dd[f'exp_wgt_return_{N}m'] = (dd['adj_close'].pct_change() * dd['turnover_rate']).rolling(21 * N).apply(
            exp_func)

    dd['ln_capital'] = np.log(dd['total_mv'])
    dd['month'] = dd['datetime'].apply(lambda x: x[:6])
    dd = dd.sort_values(by=['datetime']).reset_index(drop=True)
    dd = dd.drop(['datetime'], axis=1)
    dd = dd.drop_duplicates(subset=['month'], keep='last')
    return dd


def factors3(code):
    dd_index = pro.index_daily(ts_code='000001.SH')
    dd_index.rename(columns={'trade_date': 'datetime'}, inplace=True)
    dd_index = dd_index.sort_values(by=['datetime']).reset_index(drop=True)
    dd_code = get_adj_data(code=code, freq='daily')
    dd_code.rename(columns={'trade_date': 'datetime'}, inplace=True)
    dd_code['datetime'] = dd_code['datetime'].astype(str)
    dd_index['datetime'] = dd_index['datetime'].astype(str)
    dd = pd.merge(dd_code[['datetime', 'close']], dd_index[['datetime', 'close']], on=['datetime'], how='right').fillna(
        method='ffill').dropna()
    dd.columns = ['datetime', 'code', 'index']
    dd = dd.sort_values(by=['datetime']).reset_index(drop=True)
    dd['code_ret'] = dd['code'].pct_change()

    dd['ln_price'] = np.log(dd['code'])

    for N in [1, 3, 6, 12]:
        dd[f'std_{N}m'] = dd['code_ret'].rolling(21 * N).std()

    dd['DIF'] = dd['code'].ewm(span=15).mean() - dd['code'].ewm(span=30).mean()
    dd['DEA'] = dd['DIF'].ewm(span=10).mean()
    dd['MACD'] = dd['DIF'] - dd['DEA']

    dd['RSI'] = 100 * pd.Series(SMA(np.maximum(dd['code'] - dd['code'].shift(1), 0), 20, 1)) / pd.Series(
        SMA(np.abs(dd['code'] - dd['code'].shift(1)), 20, 1))
    dd['PSY'] = dd['code_ret'].rolling(20).apply(lambda x: len(np.where(x.values > 0)[0]) / 20)
    dd['BIAS'] = (dd['code'] - dd['code'].rolling(20).mean()) / (dd['code'].rolling(20).mean())
    dd['month'] = dd['datetime'].apply(lambda x: x[:6])
    dd = dd.sort_values(by=['datetime']).reset_index(drop=True)
    dd = dd.drop(['datetime', 'code', 'index', 'code_ret'], axis=1)
    dd = dd.drop_duplicates(subset=['month'], keep='last')
    return dd


def factors4(folder, code):
    file = f'{code}.csv'
    dd = pd.read_csv(f'{folder}/{file}')
    dd['datetime'] = dd['datetime'].astype(str)
    dd = addfield(dd, code)

    # 自有的或简单计算即可获得的因子
    Sales_G_q = 'or_yoy'
    Profit_G_q = 'netprofit_yoy'
    OCF_G_q = 'cfps_yoy'
    ROE_G_q = 'roe_yoy'

    dd['EP'] = 1 / dd['pe_ttm']
    dd['EPcut'] = dd['profit_dedt'] / dd['total_mv']
    dd['BP'] = 1 / dd['pb']
    # SP=营业总收入/总市值=1/((净利润/营业总收入)*(总市值/净利润))=1/(profit_to_gr*pe_ttm)
    dd['SP'] = (dd['profit_to_gr'] * dd['pe_ttm']) / 100
    dd['NCFP'] = dd['total_share'] * dd['cfps'] / dd['total_mv']
    dd['OCFP'] = dd['total_share'] * dd['ocfps'] / dd['total_mv']
    dd['DP'] = dd['dv_ttm'] / dd['total_mv']
    dd['G/PE'] = dd['q_profit_yoy'] / dd['pe_ttm']

    # 杠杆因子
    dd['financial_leverage'] = dd['assets_to_eqt']
    dd['current_ratio'] = dd['current_ratio']
    dd['debtequityratio'] = dd['total_ncl'] / (dd['total_assets'] - dd['total_liab'])
    dd['cashratio'] = dd['cash_ratio']

    ROE_q = 'roe'
    ROE_ttm = 'roe_yearly'
    ROA_q = 'roa'
    ROA_ttm = 'roa_yearly'
    grossprofitmargin_q = 'q_gsprofit_margin'
    grossprofitmargin_ttm = 'grossprofit_margin'

    # 扣除非经常损益后的净利润/净利润 * 净利润/营业总收入 = 扣除非经常损益后的净利润/营业总收入 = dtprofit_to_profit*profit_to_gr
    dd['profitmargin_q'] = dd['dtprofit_to_profit'] * dd['q_profit_to_gr']
    dd['profitmargin_ttm'] = dd['dtprofit_to_profit'] * dd['profit_to_gr']

    dd['assetturnover_ttm'] = dd['assets_turn']  # 季度的资产周转率找不到
    dd['operationcashflowratio_q'] = dd['ocfps'] / dd['q_npta']
    dd['operationcashflowratio_ttm'] = dd['ocfps'] / dd['npta']

    成长因子 = [Sales_G_q, Profit_G_q, OCF_G_q, ROE_G_q]
    估值因子 = ['EP', 'EPcut', 'BP', 'SP', 'NCFP', 'OCFP', 'DP', 'G/PE']
    杠杆因子 = ['financial_leverage', 'current_ratio', 'debtequityratio', 'cashratio']
    财务质量因子 = [ROE_q, ROE_ttm, ROA_q, ROA_ttm, grossprofitmargin_q, grossprofitmargin_ttm,
              'profitmargin_q', 'profitmargin_ttm', 'assetturnover_ttm', 'operationcashflowratio_q',
              'operationcashflowratio_ttm']

    dd = dd[['datetime'] + 成长因子 + 估值因子 + 杠杆因子 + 财务质量因子]
    dd['month'] = dd['datetime'].apply(lambda x: x[:6])
    dd = dd.sort_values(by=['datetime'])
    dd = dd.drop(['datetime'], axis=1)
    dd = dd.drop_duplicates(subset=['month'], keep='last')
    return dd


def get_all_factors(folder, code, start_date, end_date):
    # 计算HAlpha,beta,return_Nm
    # 直接基于月频数据行情数据计算，没有财务，没有日频
    dd1 = factors1(code)

    # 计算turn_Nm,bias_turn_Nm,wgt_return_Nm,exp_wgt_return_Nm
    # 计算各种换手率相关的因子以及市值因子
    dd2 = factors2(folder, code)

    # 计算ln_price,std_Nm,DIF,DEA,MACD,RSI,PSY,BIAS
    # 不涉及到保存的基础数据文件夹中数据的日频行情因子
    dd3 = factors3(code)

    # 计算各种财务类的数据，会用到之前准备的基础数据
    dd4 = factors4(folder, code)

    if len(dd1) > 0 and len(dd2) > 0 and len(dd3) > 0 and len(dd4) > 0:
        ddd = pd.merge(dd1, dd2, on=['month'])
        ddd = pd.merge(ddd, dd3, on=['month'])
        ddd = pd.merge(ddd, dd4, on=['month'])
        ddd['datetime'] = ddd['datetime'].astype(str)
        ddd = ddd[(ddd['datetime'] >= start_date) & (ddd['datetime'] <= end_date)].reset_index(drop=True)
        ddd.index = pd.to_datetime(ddd['datetime'])
        ddd = ddd.drop(['datetime', 'month'], axis=1)
    else:
        ddd = pd.DataFrame()
    return ddd


def de_extreme(ddd):
    """
    中位数去极值
    进来的和出去的df都必须是纯数值，因此时间或日期列必须弃掉，用索引装载
    """
    DM = ddd.median()
    DM1 = (ddd - DM).abs().median()
    上限 = DM + 5 * DM1
    下限 = DM - 5 * DM1
    ddd = (ddd > 上限) * 上限 + (ddd < 下限) * 下限 + ((ddd >= 下限) & (ddd <= 上限)) * ddd
    return ddd
