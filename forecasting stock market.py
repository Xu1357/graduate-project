#########################根据原始数据构造变量####################################

import numpy as np
import pandas as pd
from datetime import datetime
import os


path = os.getcwd()
raw_data_path = os.path.join(path, "微调后原始数据")  #原始数据的路径
dependent_path = os.path.join(path, "因变量")  #因变量路径
independent_path = os.path.join(path, "自变量")  #自变量路径


##构造因变量，注意这里市值的单位是千元
TRD_Cnmont = pd.read_csv(os.path.join(raw_data_path, "TRD_Cnmont.csv"), index_col="Trdmnt")  #月度收益率、市值
TRD_Cnmont.index = pd.to_datetime(TRD_Cnmont.index)
TRD_Nrrate = pd.read_csv(
    os.path.join(raw_data_path, "TRD_Nrrate" + ".csv"),
    index_col=1
)   #无风险利率
TRD_Nrrate.index = pd.to_datetime(TRD_Nrrate.index)
TRD_Cnmont = pd.concat(
    [TRD_Cnmont, TRD_Nrrate["Nrrmtdt"] / 100],  #无风险利率是百分数
    axis=1,
    join="inner"
)
dependent1 = TRD_Cnmont["Cmretwdtl"] - TRD_Cnmont["Nrrmtdt"]  #因变量1为总市值加权收益-无风险利率
dependent2 = TRD_Cnmont["Cmretwdos"] - TRD_Cnmont["Nrrmtdt"]  #因变量2为流通市值加权收益-无风险利率
dependent = pd.DataFrame({
    "dependent1": dependent1,
    "dependent2": dependent2
})
dependent.to_csv(os.path.join(dependent_path, "dependent.csv"))
dependent = pd.read_csv(os.path.join('因变量', 'dependent.csv'), index_col=0)
dependent.index = pd.to_datetime(dependent.index)

##构造D12，单位为元
CD_Dividend = pd.read_csv(os.path.join(raw_data_path, "CD_Dividend.csv"), index_col=1).dropna()
CD_Dividend.index = pd.to_datetime(CD_Dividend.index)
CD_Dividend["year"] = [x.year for x in CD_Dividend.index]
CD_Dividend["month"] = [x.month for x in CD_Dividend.index]
monthly_dividend = pd.DataFrame(CD_Dividend.groupby(["year", "month"]).sum())
monthly_dividend.reset_index(inplace=True)
monthly_dividend = monthly_dividend[monthly_dividend["year"] >= 1995]
monthly_dividend.to_csv(os.path.join(independent_path, "dividend_temp.csv"))  #保存至本地并手动补全月份更简单
monthly_dividend = pd.read_csv(os.path.join(independent_path, "dividend_temp.csv"))
D12 = monthly_dividend["Numdiv"].rolling(window=12).sum().iloc[12:]  #结果为Series，单位为元
pd.DataFrame(D12).to_csv(os.path.join(independent_path, "D12.csv"))

##构造E12，单位为元
earnings = pd.read_csv(os.path.join(raw_data_path, "earnings.csv"), index_col=0).dropna()
earnings.index = pd.to_datetime(earnings.index)
single_earnings = earnings - earnings.shift(1)
replace = [x for x in single_earnings.index if (x.year <= 2001 and x.month == 6) or (x.year > 2001 and x.month == 3)]
single_earnings.loc[replace] = earnings.loc[replace]  #构造出了半年或单季度盈利，单位为元
E12_tempty = pd.DataFrame(
    {"E12": [0] * 300, "net_E12": [0] * 300},
    index=pd.date_range(start="1995-01-01", end="2020-01-01", freq="M")
)  #创建一个空的月度E12用于填充

for m in E12_tempty.index:
    if m.year <= 2001:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 6, 30)]) / 6
        elif m.month > 3 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 6
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 6
    elif m.year == 2002:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 6, 30)]) / 6
        elif m.month > 3 and m.month <= 6:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 6
        elif m.month > 6 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 3, 31)]) / 3
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 3
    else:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 9, 30)]) / 3
        elif m.month > 3 and m.month <= 6:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 3
        elif m.month > 6 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 3, 31)]) / 3
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 3

E12_tempty = E12_tempty.rolling(window=12).sum().loc[datetime(1996, 1, 31):]
E12_tempty.to_csv(os.path.join(independent_path, "E12.csv"))
E12 = E12_tempty["E12"].tolist()
net_E12 = E12_tempty["net_E12"].tolist()

##构造D/E，结果为np.array
D_E1 = np.log(D12.tolist()) - np.log(E12)
D_E2 = np.log(D12.tolist()) - np.log(net_E12)
pd.DataFrame(
    {"D/E1": D_E1, "D/E2": D_E2},
    index=TRD_Cnmont.index
).to_csv(os.path.join(independent_path, "D_E.csv"))

D_E = pd.read_csv(os.path.join('自变量', 'D_E.csv'), index_col=0)
D_E.index = pd.to_datetime(D_E.index)


##构造D/P，结果为np.array
D_P = np.log(D12.tolist()) - np.log((TRD_Cnmont["Cmmvttl"] * 1000).tolist())
pd.DataFrame(
    {"D/P": D_P},
    index=TRD_Cnmont.index
).to_csv(os.path.join(independent_path, "D_P.csv"))

D_P = pd.read_csv(os.path.join('自变量', 'D_P.csv'), index_col=0)
D_P.index = pd.to_datetime(D_P.index)


##构造D/Y，结果为np.array
Y = pd.read_csv(os.path.join(raw_data_path, "TRD_Cnmont" + ".csv"), index_col="Trdmnt")["Cmmvttl"]  #重新导入市值
D_Y = np.log(D12.tolist()) - np.log((Y.shift(1)[1:] * 1000).tolist())
pd.DataFrame(
    {"D/Y": D_Y},
    index=TRD_Cnmont.index
).to_csv(os.path.join(independent_path, "D_Y.csv"))

D_Y = pd.read_csv(os.path.join('自变量', 'D_Y.csv'), index_col=0)
D_Y.index = pd.to_datetime(D_Y.index)


##构造E/P，结果为np.array
E1_P = np.log(E12) - np.log((TRD_Cnmont["Cmmvttl"] * 1000).tolist())
E2_P = np.log(net_E12) - np.log((TRD_Cnmont["Cmmvttl"] * 1000).tolist())
pd.DataFrame(
    {"E1/P": E1_P, "E2/P": E2_P},
    index=TRD_Cnmont.index
).to_csv(os.path.join(independent_path, "E_P.csv"))

E_P = pd.read_csv(os.path.join('自变量', 'E_P.csv'), index_col=0)
E_P.index = pd.to_datetime(E_P.index)


##填充账面价值，单位为元
book_value = pd.read_csv(os.path.join(raw_data_path, "book_value.csv"), index_col=0).dropna()
book_value.index = pd.to_datetime(book_value.index)
book_tempty = pd.DataFrame(
    {"book_value": [0] * 288},
    index=TRD_Cnmont.index
)

for m in book_tempty.index:
    if m.year <= 2001:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 6, 30)])
        elif m.month > 3 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 31)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 30)])
    elif m.year == 2002:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 6, 30)])
        elif m.month > 3 and m.month <= 6:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 31)])
        elif m.month > 6 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 3, 31)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 30)])
    else:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 9, 30)])
        elif m.month > 3 and m.month <= 6:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 31)])
        elif m.month > 6 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 3, 31)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 30)])

book_tempty.to_csv(os.path.join(independent_path, "book_value.csv"))
book = book_tempty["book_value"].tolist()   #填充后的月度账面价值，是list


##构造B/M，结果为np.array
B_M = np.array(book) / np.array(TRD_Cnmont["Cmmvttl"] * 1000)
pd.DataFrame(
    {"B/M": B_M},
    index=TRD_Cnmont.index
).to_csv(os.path.join(independent_path, "B_M.csv"))

B_M = pd.read_csv(os.path.join('自变量', 'B_M.csv'), index_col=0)
B_M.index = pd.to_datetime(B_M.index)


##构造SVAR，结果为np.array，且包含2020年1月的数据
TRD_Cndalym = pd.read_csv(os.path.join(raw_data_path, "TRD_Cndalym.csv"), index_col="Trddt")
TRD_Cndalym = TRD_Cndalym[TRD_Cndalym["Markettype"] == 5]  #筛选出综合A股市场收益率
TRD_Cndalym.index = pd.to_datetime(TRD_Cndalym.index)

daily_ret1 = TRD_Cndalym["Cdretwdtl"]
daily_ret2 = TRD_Cndalym["Cdretwdos"]
daily_ret = pd.DataFrame(
    {"ret1": daily_ret1, "ret2": daily_ret2}
)
daily_ret["year"] = [x.year for x in daily_ret.index]
daily_ret["month"] = [x.month for x in daily_ret.index]
SVAR = daily_ret.groupby(["year", "month"]).apply(lambda x: (x ** 2).sum())[["ret1", "ret2"]]
SVAR.to_csv(os.path.join(independent_path, "SVAR.csv"))
SVAR1 = np.array(SVAR["ret1"])
SVAR2 = np.array(SVAR["ret2"])
# 把2020年1月的数据删除了再read
SVAR = pd.read_csv(os.path.join('自变量', 'SVAR.csv'))[['ret1', 'ret2']]
SVAR.columns = ['SVAR1', 'SVAR2']
SVAR.index = B_M.index



##构造INF，结果为np.array
INF = pd.read_csv(os.path.join(raw_data_path, "INF.csv"), index_col=0)
INF.index = pd.to_datetime(INF.index)
INF = INF.shift(1).dropna() / 100
INF.columns = ['INF']


##构造新股发行量的移动加总，单位为股，类型为np.array
IPO = pd.read_csv(os.path.join(raw_data_path, "IPO.csv"), index_col=0)
IPO.index = pd.to_datetime(IPO.index)
IPO["year"] = [x.year for x in IPO.index]
IPO["month"] = [x.month for x in IPO.index]
IPO = IPO[IPO["year"] >= 1995]
IPO = IPO[IPO["year"] <= 2019]
IPO.reset_index(inplace=True)
IPO.sort_values(by="IPO_date", inplace=True)
IPO.set_index("IPO_date", inplace=True)
monthly_IPO = IPO.groupby(["year", "month"]).sum()  #每个月的IPO
monthly_IPO.reset_index(inplace=True)
monthly_IPO.index = [datetime(monthly_IPO["year"][i], monthly_IPO["month"][i], 1) for i in range(len(monthly_IPO))]
IPO_tempty = pd.DataFrame(
    [0] * 300,
    index=pd.date_range(start="1995-01-01", end="2019-12-01", freq="MS")
)
full_monthly_IPO = pd.concat(
    [IPO_tempty, monthly_IPO],
    axis=1,
    join="outer"
)["IPO_number"].fillna(0)
pd.DataFrame(full_monthly_IPO).to_csv(os.path.join(independent_path, "IPO.csv"))
full_monthly_IPO = pd.read_csv(os.path.join('自变量', 'IPO.csv'), index_col=0)
full_monthly_IPO.index = pd.to_datetime(full_monthly_IPO.index)
IPO_sum = full_monthly_IPO.rolling(window=12).sum().loc[datetime(1996, 1, 1):]


##构造NTIS，类型为np.array，发行量的单位为股
NTIS = IPO_sum.values[0] / TRD_Cnmont["Cmmvttl"].values
NTIS = pd.Series(NTIS, index=INF.index)
NTIS.name = 'NTIS'


##构造TO，类型为np.array，总交易量的单位为股
TV_Mont = pd.read_csv(os.path.join(raw_data_path, "TV_Mont.csv"), index_col=1)
SH_volume = TV_Mont[TV_Mont["Markettype"] == 1]
SZ_volume = TV_Mont[TV_Mont["Markettype"] == 4]
volume = SH_volume["Mnshrtrdtl"] + SZ_volume["Mnshrtrdtl"]
pd.DataFrame({"volume": volume}).to_csv(os.path.join(independent_path, "volume.csv"))
TO = np.array(volume) / np.array(TRD_Cnmont["Cmmvttl"])
TO = pd.Series(TO, index=INF.index)
TO.name = 'TO'


##构造货币相关变量
CMMPI_Mam = pd.read_csv(os.path.join(raw_data_path, "CMMPI_Mam.csv"), index_col=0)
M2 = CMMPI_Mam["Mam01"][1:]
M2G = M2 / M2.shift(1) - 1
M2G = M2G.dropna()
M0 = CMMPI_Mam["Mam03"][1:]
M0G = M0 / M0.shift(1) - 1
M0G = M0G.dropna()
M1 = CMMPI_Mam["Mam02"]
M1G = (M1 / M1.shift(1) - 1).dropna()
M1G = (M1G - M1G.shift(1)).dropna()
pd.DataFrame(
    {"M0G": M0G, "M1G": M1G, "M2G": M2G},
    index=pd.date_range(start="1996-01-01", end="2019-11", freq="MS")
).to_csv(os.path.join(independent_path, "M.csv"))
M = pd.read_csv(os.path.join('自变量', 'M.csv'), index_col=0)
M.index = pd.to_datetime(M.index)


##整合所有变量
pd.DataFrame(
    {
        "dependent1": dependent1.tolist()[1:],
        "dependent2": dependent2.tolist()[1:],
        "D/E1": D_E1[1:],
        "D/E2": D_E2[1:],
        "D/P": D_P[1:],
        "D/Y": D_Y[1:],
        "E1/P": E1_P[1:],
        "E2/P": E2_P[1:],
        "B/M": B_M[1:],
        "SVAR1": SVAR1[1:len(SVAR1) - 1],
        "SVAR2": SVAR2[1:len(SVAR2) - 1],
        "INF": INF[1:],
        "NTIS": NTIS[1:],
        "TO": TO[1:],
        "M0G": M0G,
        "M1G": M1G,
        "M2G": M2G
    },
    index=pd.date_range(start="1996-02-01", end="2019-12-01", freq="MS")
).to_csv("all_data.csv")
pd.concat(
    [
        dependent,
        D_E,
        D_P,
        D_Y,
        E_P,
        B_M,
        SVAR,
        INF,
        NTIS,
        TO,
        M
    ],
    axis=1,
    join='outer'
).to_csv('all_data.csv')




########################################预测市场######################################
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import os
from scipy import stats
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, LarsCV, OrthogonalMatchingPursuitCV
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

path = os.getcwd()
merge_data = pd.read_csv("all_data.csv", index_col=0)
merge_data.index = pd.to_datetime(merge_data.index)

combination = {
    "kind1": [0] + list(range(3, 6)) + list(range(7, 10)) + list(range(11, 17)),
    "kind2": [0, 2] + list(range(4, 7)) + [8, 9] + list(range(11, 17)),
    "kind3": [1, 2] + list(range(4, 7)) + [8, 9] + list(range(11, 17)),
    "kind4": [1] + list(range(3, 6)) + list(range(7, 10)) + list(range(11, 17))
}  #所有组合情况


all_months = merge_data.index.tolist()  #所有月份
#所有样本外月份
start_month = datetime(2002, 1, 1)  #样本外开始月份
out_months = list(merge_data.loc[start_month:, ].index)  #所有样本外月份
out_num = len(out_months)  #样本外月份数

#组合一
data1 = merge_data.iloc[:, combination["kind1"]]
#样本外真实收益
true_ret = data1.iloc[:, 0].iloc[all_months.index(start_month):]
# 数据的描述性统计
Des = data1.describe().T
Des['自相关系数'] = data1.apply(lambda x: x.corr(x.shift(1)))
Des.to_excel('描述性统计.xlsx')


#找训练集
def window(data, kind, mon, m=10, w_min=24):
    '''
    对指定要预测收益的月份，根据窗口方式输出训练数据集

    Parameters
    ----------
    data: 原始数据集  (pd.DataFrame)
    kind: 窗口类型  (str  "expanding" or "Avew")
    mon: 指定的月份  (datetime)
    m: Avew方法划分参数m  (int default 10)
    w_min: Avew方法划分参数w_min  (int default 24)

    Returns
    -------
    训练数据集  (pd.DataFrame for "expanding", list of pd.DataFrame for "Avew")
    '''
    if kind == "expanding":
        pos = all_months.index(mon)
        train_df = data.iloc[:pos]   #训练集
    else:
        pos = all_months.index(mon)
        #W = data.iloc[:pos]  #Avew方法输入的观测窗口
        w_i = [w_min + (i - 1) / (m - 1) * (pos - w_min) for i in range(1, m + 1)]
        w_i = [int(x) for x in w_i]
        train_df = [data.iloc[(pos - x - 1) : pos] for x in w_i[:-1]]  #训练集
        train_df.append(data.iloc[:pos])

    return train_df


def historical_mean(data=data1, window_type="expanding", m=10, w_min=24):
    '''
    采用历史均值方法预测

    Parameters:
    -----------
    data: 原始数据集  (pd.DataFrame)
    window_type: 训练集选择方式  (str  "expanding" or "Avew"  default "expanding")
    m: Avew方法划分参数m  (int default 10)
    w_min: Avew方法划分参数w_min  (int default 24)

    Returns:
    --------
    采用历史均值方法预测出的收益时间序列  (pd.Series)
    '''
    #预测
    prediction = []
    for mon in out_months:
        if window_type == "expanding":
            temp = window(data=data, kind=window_type, mon=mon, m=m, w_min=w_min)
            pre_ret = temp.iloc[:, 0].mean()  #收益率预测
            prediction.append(pre_ret)
        else:
            temp = window(data=data, kind=window_type, mon=mon, m=m, w_min=w_min)
            pre_ret = np.mean([x.iloc[:, 0].mean() for x in temp])
            prediction.append(pre_ret)

    prediction = pd.Series(prediction, index=out_months)
    
    return prediction


hist_mean1 = historical_mean()  #expanding窗口的预测
hist_mean2 = historical_mean(window_type="Avew")  #Avew窗口的预测
pd.DataFrame({
    "expanding": hist_mean1,
    "Avew": hist_mean2
}).to_csv(
    os.path.join(path, "预测结果", "历史均值预测.csv")
)
hist_mean1 = pd.read_csv(os.path.join("预测结果", "历史均值预测.csv"), index_col=0)["expanding"]
hist_mean2 = pd.read_csv(os.path.join("预测结果", "历史均值预测.csv"), index_col=0)["Avew"]


#单变量预测
def single_variable(data=data1):
    '''
    Welch and Goyal (2008) 的单变量预测

    Parameters
    ----------
    data: 原始数据集  (pd.DataFrame)

    Returns
    -------
    预测出的收益时间序列  (pd.DataFrame)
    '''
    variables = data.columns[1:]  #用于预测的自变量
    all_pred = []  #用于存储不同截面的预测值
    for month in out_months:
        temp0 = window(data=data, kind="expanding", mon=month)  #获取训练集
        pred_i = []  #该截面的预测数据
        for v in variables:
            temp1 = temp0[v]  #提取该变量的数据
            temp2 = temp0.iloc[:, 0]  #提取收益数据
            result = sm.OLS(temp2.iloc[1:].tolist(), sm.add_constant(temp1.iloc[:-1].tolist())).fit()  #拟合
            pred_i.append(result.predict([1, temp1.iloc[-1]])[0])  #预测
        all_pred.append(pred_i)
    all_pred = pd.DataFrame(all_pred, index=out_months, columns=variables)

    return all_pred

prediction1 = single_variable()
prediction1.to_csv(os.path.join(path, "预测结果", "单变量预测.csv"))
prediction1 = pd.read_csv(
    os.path.join(path, "预测结果", "单变量预测.csv"),
    index_col=0
)


#样本外R方及CW-test
def OOS_R2(prediction, benchmark=hist_mean1):
    '''
    计算样本外R方及对应的CW-test

    Parameters
    ----------
    prediction: 超额收益预测的时间序列  (pd.Series)
    benchmark: 比较的基准  (pd.Series)

    Returns:
    --------
    样本外R方  (float)
    CW-test统计量  (float)
    '''
    MSFE_M = ((prediction - true_ret) ** 2).mean()  #预测值的MSFE
    MSFE_bmk = ((benchmark - true_ret) ** 2).mean()  #基准的MSFE
    R2_OOS = 1 - MSFE_M / MSFE_bmk
    #CW-test
    f = (true_ret - benchmark) ** 2 - (true_ret - prediction) ** 2 + (benchmark - prediction) ** 2
    t_stat = stats.ttest_1samp(f, 0)[0]

    return R2_OOS, t_stat


#单变量预测的统计检验
def R2_test(prediction, name):
    '''
    预测结果的统计检验，包含R方，结果储存至本地

    Parameters
    ----------
    prediciton: 预测结果  (pd.DataFrame)
    name: 文件名，含扩展名  (str)
    '''
    result = prediction.apply(OOS_R2, benchmark=hist_mean1)
    pd.DataFrame(
        {"R_square": [x[0] for x in result], "CW-test": [x[1] for x in result]},
        index=prediction.columns
    ).to_csv(
        os.path.join(path, "统计检验", "R方", name)
    )

R2_test(prediction1, "单变量预测.csv")



#多元回归预测
#一个数据集的多元回归预测
def MR_i(data):
    '''
    一个数据集的多元回归预测

    Parameters
    ----------
    data: 给定的数据集  (pd.DataFrame)

    Returns
    -------
    预测结果  (float)
    '''
    Y = data.iloc[1:, 0]  #收益率观测序列
    X = data.iloc[:, 1:]  #所有变量观测值
    model = sm.OLS(Y.values, sm.add_constant(X.iloc[:-1, :].values)).fit()  #拟合
    
    return model.predict([1] + X.iloc[-1].tolist())[0]


def MR(data=data1, window_type="expanding", m=10, w_min=24):
    '''
    多元回归预测

    Parameters
    ----------
    data: 原始数据  (pd.DataFrame)
    window_type: 窗口类型  (str  "expanding" or "Avew")
    m: Avew窗口的参数  (int)
    w_min: Avew窗口的参数  (int)

    Returns
    -------
    预测结果  (pd.Series)
    '''
    all_pred = []
    for mon in out_months:
        train = window(data=data, mon=mon, kind=window_type, m=m, w_min=w_min)  #获取训练集
        if window_type == "expanding":
            all_pred.append(MR_i(train))
        else:
            all_pred.append(np.mean([MR_i(x) for x in train]))
        
    return pd.Series(all_pred, index=out_months)

prediction3_expanding = MR()
prediction3_Avew = MR(window_type="Avew")
pd.DataFrame({
    "expanding": prediction3_expanding,
    "Avew": prediction3_Avew
}).to_csv(os.path.join(path, "预测结果", "多元回归预测.csv"))
MR_pre = pd.read_csv(os.path.join(
    path,
    "预测结果",
    "多元回归预测.csv"
), index_col=0)





#线性框架下的预测
#一个数据集的线性预测
def linear_i(data, linear_func):
    '''
    一个数据集的线性机器学习预测

    Parameters
    ----------
    data: 给定的数据集  (pd.DataFrame)
    linear_func: 线性模型  (function)

    Returns
    -------
    预测结果  (float)
    '''
    Y = data.iloc[1:, 0]  #收益率的训练集
    X = data.iloc[:, 1:]  #自变量的序列
    model = linear_func.fit(X.iloc[:-1, :].values, Y.values)  #拟合

    return model.predict([X.iloc[-1].tolist()])[0]

def linear_prediction(linear_func, data=data1, window_type="expanding", m=10, w_min=24):
    '''
    线性预测

    Parameters
    ----------
    linear_func: 线性模型  (function)
    data: 原始数据  (pd.DataFrame)
    window_type: 窗口类型  (str  "expanding" or "Avew")
    m: Avew窗口的参数  (int)
    w_min: Avew窗口的参数  (int)
    '''
    all_pred = []
    for mon in out_months:
        train = window(data=data, mon=mon, kind=window_type, m=m, w_min=w_min)  #获取训练集
        if window_type == "expanding":
            all_pred.append(linear_i(train, linear_func=linear_func))
        else:
            all_pred.append(np.mean([linear_i(x, linear_func=linear_func) for x in train]))
        
    return pd.Series(all_pred, index=out_months)

#广义交叉验证的岭回归
ridge_pre_expanding = linear_prediction(RidgeCV())
#五折交叉验证的Lasso回归
lasso_pre_expanding = linear_prediction(LassoCV(cv=5))
#五折交叉验证的弹性网络
elasticnet_pre_expanding = linear_prediction(ElasticNetCV(cv=5))
#五折交叉验证的最小角回归
lars_pre_expanding = linear_prediction(LarsCV(cv=5))
#五折交叉验证的正交匹配追踪
omp_pre_expanding = linear_prediction(OrthogonalMatchingPursuitCV(cv=5))


#保存结果
pd.DataFrame({
    "ridge": ridge_pre_expanding,
    "lasso": lasso_pre_expanding,
    "elastic net": elasticnet_pre_expanding,
    "lars": lars_pre_expanding,
    "OMP": omp_pre_expanding
}).to_csv(
    os.path.join(path, "预测结果", "expanding方式窗口预测.csv")
)
linear_pre_expanding = pd.read_csv(
    os.path.join(
        path,
        "预测结果",
        "expanding方式窗口预测.csv"
    ),
    index_col=0
)


###改变窗口方式之后的预测
ridge_pre_Avew = linear_prediction(RidgeCV(), window_type="Avew")
lasso_pre_Avew = linear_prediction(LassoCV(cv=5), window_type="Avew")
elasticnet_pre_Avew = linear_prediction(ElasticNetCV(cv=5), window_type="Avew")
lars_pre_Avew = linear_prediction(LarsCV(cv=5), window_type="Avew")
omp_pre_Avew = linear_prediction(OrthogonalMatchingPursuitCV(cv=5), window_type="Avew")
pd.DataFrame({
    "ridge": ridge_pre_Avew,
    "lasso": lasso_pre_Avew,
    "elastic net": elasticnet_pre_Avew,
    "lars": lars_pre_Avew,
    "OMP": omp_pre_Avew
}).to_csv(
    os.path.join(path, "预测结果", "Avew方式窗口预测.csv")
)
linear_pre_Avew = pd.read_csv(
    os.path.join(
        path,
        "预测结果",
        "Avew方式窗口预测.csv"
    ),
    index_col=0
)

FC_pre_expanding = linear_pre_expanding.mean(axis=1)
FC_pre_Avew = linear_pre_Avew.mean(axis=1)
pd.DataFrame({
    "expanding": FC_pre_expanding,
    "Avew": FC_pre_Avew
}).to_csv(os.path.join(path, "预测结果", "FC.csv"))
FC_pre = pd.read_csv(os.path.join(
    path,
    "预测结果",
    "FC.csv"
), index_col=0)

#综合结果
##expanding方式
linear_pre_expanding["KitchenSink"] = MR_pre["expanding"]
linear_pre_expanding['FC'] = FC_pre['expanding']
linear_pre_expanding = linear_pre_expanding[[
    'KitchenSink',
    'ridge',
    'lasso',
    'elastic net',
    'lars',
    'OMP',
    'FC'
]]
linear_pre_expanding.to_csv(os.path.join(
    path,
    "预测结果",
    "expanding方式综合结果.csv"
))
all_linear_expanding = pd.read_csv(
    os.path.join(
        path,
        "预测结果",
        "expanding方式综合结果.csv"
    ),
    index_col=0
)

##Avew方式
linear_pre_Avew["KitchenSink"] = MR_pre["Avew"]
linear_pre_Avew['FC'] = FC_pre['Avew']
linear_pre_Avew = linear_pre_Avew[[
    'KitchenSink',
    'ridge',
    'lasso',
    'elastic net',
    'lars',
    'OMP',
    'FC'
]]
linear_pre_Avew.to_csv(os.path.join(
    path,
    "预测结果",
    "Avew方式综合结果.csv"
))
all_linear_Avew = pd.read_csv(
    os.path.join(
        path,
        "预测结果",
        "Avew方式综合结果.csv"
    ),
    index_col=0
)

##DCSFE时序图
def DCSFE_plot(prediction):
    '''
    作出DCSFE时序图

    Parameters
    ----------
    prediction: 预测结果  (pd.DataFrame)
    '''
    hist_mean_FE = true_ret - hist_mean1  #历史均值预测的误差
    hist_mean_CSFE = (hist_mean_FE ** 2).cumsum()  #历史均值的累计平方误差
    prediction_FE = prediction.apply(lambda x: true_ret - x)  #预测模型的误差
    prediction_CSFE = (prediction_FE ** 2).cumsum()  #预测模型的累计平方误差

    #作图
    temp = prediction_CSFE.apply(lambda x: hist_mean_CSFE - x)
    for i in range(1, 13):
        plt.subplot(4, 3, i)
        plt.plot(temp.index, temp.iloc[:, i - 1], linestyle='dashed')
        plt.title(temp.columns[i - 1])
        plt.ylim(-0.1, 0.1)
        plt.xlim(datetime(2002, 1, 1), xmax=datetime(2019, 12, 1))
        plt.hlines(y=0, xmin=datetime(2002, 1, 1), xmax=datetime(2019, 12, 1))

DCSFE_plot(prediction1)


#expanding方式综合结果的R2
R2_test(prediction=all_linear_expanding, name="expanding方式综合结果.csv")  
#Avew方式综合结果的R2
R2_test(prediction=all_linear_Avew, name="Avew方式综合结果.csv")


#和历史平均的组合
def HA_shrinkage(prediction, factor=0.7):
    '''
    和历史平均预测的组合，使用简单加权平均

    Parameters
    ----------
    prediction: 预测结果  (pd.Series)
    factor: 收缩因子，即预测结果项的系数  (float default 0.7)

    Returns
    -------
    组合之后的预测结果  (pd.Series)
    '''
    result = prediction * factor + hist_mean1 * (1 - factor)

    return result


#预测的shrinkage
all_linear_Avew.apply(HA_shrinkage).to_csv(
    os.path.join(
        path,
        "预测结果",
        "shrinkage后的结果.csv"
    )
)

all_linear_shrinkage = pd.read_csv(
    os.path.join(
        path,
        "预测结果",
        "shrinkage后的结果.csv"
    ),
    index_col=0
)
##R2
R2_test(prediction=all_linear_shrinkage, name="shrinkage后.csv")


############utility gain################
#estimates of stock return variance
variance_hat = data1.iloc[:, 0].rolling(window=60).var().shift(1).loc[start_month:]
#risk-free rate
TRD_Nrrate = pd.read_csv(
    os.path.join(path, "微调后原始数据", "TRD_Nrrate.csv"),
    index_col=1
)
TRD_Nrrate.index = pd.to_datetime(TRD_Nrrate.index)
rf_rate = (TRD_Nrrate["Nrrmtdt"] / 100).loc[out_months]

def CER_and_Sharpe(prediction, gamma=3, weight_bound=[0, 1.5], return_sharpe=True, trans_cost=0):
    '''
    return the certainty equivalent return (and Sharpe ratio) of one prediction

    Parameters
    ----------
    prediction: prediction of stock market returns.  (pd.Series)
    gamma: relative risk aversion parameter, default 3.  (float)
    weight_bound: the weight bound on stocks,
                  weight_bound[0] is the lower bound,
                  weight_bound[1] is the upper bound.  (list)
    return_sharpe: whether to return the Sharpe ratio, default is True.  (Bool)
    trans_cost: Transaction cost ratio.  (float)
    
    Returns
    -------
    utility gains.  (float)
    Sharpe ratio if there is one.  (float)
    '''
    #weights on stocks
    weights = 1 / gamma * (prediction / variance_hat)
    #weights constrain
    weights.clip(weight_bound[0], weight_bound[1], inplace=True)
    #portfolio return
    ret = weights * true_ret + rf_rate
    #take costs to account
    ret = ret.apply(lambda x: x * (1 - trans_cost) if x >= 0 else x * (1 + trans_cost))
    #CER
    CER_M = ret.mean() - 0.5 * gamma * ret.var()

    if return_sharpe:  #if return the sharpe ratio
        Sharpe = (ret - rf_rate).mean() / ret.std()
        result = (CER_M, Sharpe)
    else:
        result = CER_M

    return result


def save_CER(prediction, name, **kw):
    '''
    Save the CER and Sharpe ratio of the prediction

    Prameters
    ---------
    prediction: Prediction of stock market returns.  (pd.DataFrame)
    name: File name, including the extension.  (str)
    **kw: Keyword parameters for function CER_and_Sharpe()
    '''
    temp = prediction.apply(CER_and_Sharpe, **kw)
    #transform the Series to df
    temp_df = pd.DataFrame(
        {
            "utility gain": [x[0] for x in temp.values],
            "Sharpe ratio": [x[1] for x in temp.values]
        },
        index=temp.index
    )
    #CER of historical mean prediction
    hist_CER = CER_and_Sharpe(hist_mean1, return_sharpe=False)
    #utility gain of prediciton
    temp_df["utility gain"] = 1200 * (temp_df["utility gain"] - hist_CER)
    #save the result
    temp_df.to_csv(os.path.join(
        path,
        "统计检验",
        "CER",
        name
    ))


#utility gain and Sharpe ratio of shrinkage results
save_CER(prediction=all_linear_Avew, name="无成本.csv", **{"trans_cost": 0})
save_CER(prediction=all_linear_Avew, name="0.5%成本.csv", **{"trans_cost": 0.005})
save_CER(prediction=all_linear_shrinkage, name="无成本（shrinkage）.csv", **{"trans_cost": 0})
save_CER(prediction=all_linear_shrinkage, name="0.5%成本（shrinkage）.csv", **{"trans_cost": 0.005})


################encompassing test###############
def encompassing_test(pre1, pre2):
    '''
    Forecast encompassing test with the null that pre1 encompasses pre2.
    
    Parameters
    ----------
    pre1: Prediction_i.  (pd.Series)
    pre2: Prediction_j.  (pd.Series)

    Returns
    -------
    P-value of the test.  (float)
    '''
    #forecast error
    FE1 = true_ret - pre1
    FE2 = true_ret - pre2
    
    g = (FE1 - FE2) * FE1
    g_bar = g.mean()
    V_hat = (1 / out_num**2) * sum((g - g_bar) ** 2)
    HLN = out_num / (out_num - 1) * (V_hat ** (-0.5)) * g_bar

    #return p-value according to t_(out_num - 1)
    return (1 - stats.t(df=(out_num-1)).cdf(HLN))

def save_encompassing_result(prediction, name):
    '''
    Save the encompassing test result of prediction.
    The entries represent the p-value of the null that column variable encompass row viable.

    Parameters
    ----------
    prediction: Prediction results of different models.  (pd.DataFrame)
    name: name: File name, including the extension.  (str)
    '''
    model_names = prediction.columns
    result = []
    for model_i in model_names:
        model_i_result = []
        for model_j in model_names:
            if model_i == model_j:
                model_i_result.append(1)
            else:
                model_i_result.append(encompassing_test(prediction[model_i], prediction[model_j]))
        
        result.append(model_i_result)

    #transform the result to df
    result = pd.DataFrame(
        result,
        index=model_names,
        columns=model_names
    ).T
    #save the result
    result.to_csv(os.path.join(
        path,
        "统计检验",
        "涵盖性检验",
        name
    ))

#add the historical mean to the prediction
all_prediction = all_linear_shrinkage.copy()
all_prediction["HA"] = hist_mean1
save_encompassing_result(all_prediction, 'shrinkage后.csv')

all_prediction = all_linear_Avew.copy()
all_prediction['HA'] = hist_mean1
save_encompassing_result(all_prediction, '不shrinkage.csv')

    


#################################robustness check###############################
#robustness check for w_min
def check_w(w=[12, 24, 36, 48, 60]):
    '''
    robustness check for w_min, save the prediction results (Avew window) and OOS R_square

    Parameters
    ----------
    w: possible w_min  (list)
    '''
    for w_min in w:
        #linear ML prediction
        pre1 = linear_prediction(RidgeCV(), w_min=w_min, window_type="Avew")
        pre2 = linear_prediction(LassoCV(cv=5), w_min=w_min, window_type="Avew")
        pre3 = linear_prediction(ElasticNetCV(cv=5), w_min=w_min, window_type="Avew")
        pre4 = linear_prediction(LarsCV(cv=5), w_min=w_min, window_type="Avew")
        pre5 = linear_prediction(OrthogonalMatchingPursuitCV(cv=5), w_min=w_min, window_type="Avew")
        pre6 = MR(w_min=w_min, window_type="Avew")
        all_pre = pd.DataFrame({
            'Kintchen Sink': pre6,
            "ridge": pre1,
            "lasso": pre2,
            "elasticnet": pre3,
            "lars": pre4,
            "OMP": pre5,
        })
        all_pre['FC'] = all_pre.iloc[:, 1:].mean(axis=1)
        #save the prediction results
        all_pre.to_csv(os.path.join(
            path,
            "稳健性检验",
            "w_min",
            "预测结果",
            "w_min=" + str(w_min) + ".csv"
        ))
        #R2 test
        R2_test(all_pre, name="w_min=" + str(w_min) + ".csv")  #then you need move the result on your own

check_w()





#robustness check for shrinkage factor
def check_shrinkage_factor(factors=np.arange(0.6, 1, 0.01)):
    '''
    Robustness check for shrinkage factors.
    Draw a plot that shows the relation between the shrinkage factor and the OOS R_square.

    Parameters
    ----------
    factors: Possible shrinkage factors.  (array-like)
    '''
    temp = []
    for f in factors:
        #shrinkage prediction
        shrinkage_pre = all_linear_Avew.apply(HA_shrinkage, factor=f)
        temp.append([x[0] for x in shrinkage_pre.apply(OOS_R2)])
    #transformed to df
    temp_df = pd.DataFrame(
        temp,
        index=factors,
        columns=all_linear_Avew.columns
    )
    temp_df.to_csv(os.path.join(
        path,
        "稳健性检验",
        "shrinkage factor",
        "关系表.csv"
    ))
    #plot
    temp_df.plot()

check_shrinkage_factor()


######################################################################################################
############################补充工作###################################
##################构造技术指标#######################
# 读取指数价格
stock_index = pd.read_csv(os.path.join('微调后原始数据', 'Index_price.csv'), index_col=0)
stock_index.index = pd.to_datetime(stock_index.index)

# 生成技术信号
def gen_tech(x, s=1, l=9, m=9, MA_=True):
    '''
    给定指标（收盘价、交易量等），根据规则生成技术信号

    Parameters
    ----------
    x: 给定的指标  (pd.Series)
    s: 移动平均的短窗口  (int, default is 1)
    l: 移动平均的长窗口  (int, default is 9)
    m: 动量规则的参数，表示当前时点前推m天  (int, default is 9)
    MA: 是根据原始指标还是根据移动平均指标生成  (bool, default is True)

    Returns
    -------
    技术信号  (pd.Series)
    '''
    if MA_:  # 移动平均规则
        MA_s = x.rolling(window=s).mean()
        MA_l = x.rolling(window=l).mean()
        result = pd.Series(
            np.where(MA_s > MA_l, 1, 0),
            index=x.index
        )
    else:  
        result = pd.Series(np.where(x > x.shift(m), 1, 0), index=x.index)
    
    return result

# 组合方式
combination1 = [(1,9), (1,12), (2,9), (2,12), (3,9), (3,12)]
# 移动平均规则的技术信号
MA = pd.DataFrame(
    [gen_tech(stock_index['Index'], s, l) for s, l in combination1],
    index=['MA' + str(x) for x in combination1]
).T
MA = MA[MA.index > datetime(1995, 12, 31)]
# 动量规则的技术信号
MOM = pd.DataFrame(
    [gen_tech(stock_index['Index'], m) for m in [9, 12]],
    index=['MOM' + str(x) for x in [9, 12]]
).T
MOM = MOM[MOM.index > datetime(1995, 12, 31)]

# 读取全A交易量
Volume = pd.read_csv(os.path.join('微调后原始数据', 'volume.csv'), index_col=0)
Volume.index = pd.to_datetime(Volume.index)
# 生成价格哑变量
D = pd.Series(
    np.where(stock_index['Index'] >= stock_index['Index'].shift(1), 1, -1),
    index=Volume.index
).iloc[1:]
# 平衡交易量
OBV = (Volume['volume'].iloc[1:] * D).cumsum()
# 基于交易量的信号
VOL = pd.DataFrame(
    [gen_tech(OBV, s, l) for s, l in combination1],
    index=['VOL' + str(x) for x in combination1]
).T
VOL = VOL[VOL.index > datetime(1995, 12, 31)]
# 保存技术信号
pd.concat(
    [MA, MOM, VOL],
    axis=1
).to_csv(os.path.join('自变量', 'TECH.csv'))

###############技术指标预测####################
TECH_ind = pd.read_csv(os.path.join('自变量', 'TECH.csv'), index_col=0)
TECH_ind.index = merge_data.index
data1 = pd.concat([merge_data.iloc[:, 0], TECH_ind], axis=1)
TECH_pre = single_variable()
TECH_pre.to_csv(os.path.join('预测结果', '技术指标单变量.csv'))
TECH_pre = pd.read_csv(os.path.join('预测结果', '技术指标单变量.csv'), index_col=0)

# R2
R2_test(TECH_pre, "技术指标单变量.csv")