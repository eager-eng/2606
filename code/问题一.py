import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def 计算小时重叠(开始时间, 结束时间, 小时):
    return max(0.0, min(结束时间, 小时 + 1) - max(开始时间, 小时))


def 构造到达需求序列(任务数据, 小时数=2400, 区域列表=None, 类型列表=None):
    if 区域列表 is None:
        区域列表 = sorted(任务数据["SourceRegion"].unique())
    if 类型列表 is None:
        类型列表 = sorted(任务数据["TaskType"].unique())
    聚合结果 = 任务数据.groupby(["ArrivalHour", "SourceRegion", "TaskType"])["GPU_Demand"].sum()
    完整索引 = pd.MultiIndex.from_product(
        [range(小时数), 区域列表, 类型列表],
        names=["ArrivalHour", "SourceRegion", "TaskType"],
    )
    return 聚合结果.reindex(完整索引, fill_value=0).unstack(["SourceRegion", "TaskType"])


def 构造日内GPU总需求(需求序列):
    区域小时需求 = 需求序列.T.groupby(level="SourceRegion").sum().T
    区域小时需求["日内小时"] = np.asarray(区域小时需求.index) % 24
    return 区域小时需求.groupby("日内小时").mean().T


def 选择预测模型(序列, 零值率阈值=0.6):
    return "SBA-Croston" if float((pd.Series(序列) == 0).mean()) >= 零值率阈值 else "ARIMA"


def SBA克罗斯顿预测(序列, 预测步数, 平滑系数=0.1):
    数值 = np.asarray(序列, dtype=float)
    非零位置 = np.flatnonzero(数值 > 0)
    if len(非零位置) == 0:
        return pd.Series(np.zeros(预测步数))
    需求估计 = 数值[非零位置[0]]
    间隔估计 = 非零位置[0] + 1
    上次位置 = 非零位置[0]
    for 位置 in 非零位置[1:]:
        间隔 = 位置 - 上次位置
        需求估计 += 平滑系数 * (数值[位置] - 需求估计)
        间隔估计 += 平滑系数 * (间隔 - 间隔估计)
        上次位置 = 位置
    预测值 = (1 - 平滑系数 / 2) * 需求估计 / max(间隔估计, 1e-12)
    return pd.Series(np.repeat(max(预测值, 0), 预测步数))


def 计算预测指标(实际值, 预测值):
    实际 = np.asarray(实际值, dtype=float)
    预测 = np.asarray(预测值, dtype=float)
    误差 = 预测 - 实际
    分母 = np.abs(实际).sum()
    return {
        "MAE": float(np.abs(误差).mean()),
        "RMSE": float(np.sqrt(np.square(误差).mean())),
        "WMAPE": float(np.abs(误差).sum() / 分母) if 分母 > 0 else float("nan"),
    }


def ARIMA预测(序列, 预测步数, 最大阶数=3, 指定阶数=None):
    数值 = pd.Series(序列, dtype=float).reset_index(drop=True)
    if 指定阶数 is None:
        try:
            差分阶数 = 0 if adfuller(数值, autolag="AIC")[1] < 0.05 else 1
        except Exception:
            差分阶数 = 1
        最优结果 = None
        最优阶数 = None
        最优AIC = np.inf
        for 自回归阶数 in range(最大阶数 + 1):
            for 移动平均阶数 in range(最大阶数 + 1):
                try:
                    结果 = ARIMA(
                        数值,
                        order=(自回归阶数, 差分阶数, 移动平均阶数),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    if np.isfinite(结果.aic) and 结果.aic < 最优AIC:
                        最优结果 = 结果
                        最优阶数 = (自回归阶数, 差分阶数, 移动平均阶数)
                        最优AIC = float(结果.aic)
                except Exception:
                    pass
        if 最优结果 is None:
            raise RuntimeError("ARIMA参数搜索失败")
    else:
        最优阶数 = tuple(指定阶数)
        最优结果 = ARIMA(
            数值,
            order=最优阶数,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        最优AIC = float(最优结果.aic)
    预测值 = pd.Series(np.maximum(np.asarray(最优结果.forecast(预测步数), dtype=float), 0))
    return {"预测值": 预测值, "阶数": 最优阶数, "AIC": 最优AIC}


def 执行基础调度(任务数据, GPU信息, 时间数据, 网络延迟, 功率映射, 截止小时=2406):
    区域列表 = GPU信息["Region"].tolist()
    区域序号 = {区域: 序号 for 序号, 区域 in enumerate(区域列表)}
    区域参数 = GPU信息.set_index("Region")
    GPU占用 = np.zeros((len(区域列表), 截止小时), dtype=float)
    AI功率 = np.zeros((len(区域列表), 截止小时), dtype=float)
    非AI表 = 时间数据.pivot_table(index="Hour", columns="Region", values="NonAI_IT_Load_MW", aggfunc="first")
    非AI表 = 非AI表.reindex(range(截止小时)).ffill().bfill().reindex(columns=区域列表)
    非AI功率 = 非AI表.to_numpy(dtype=float).T
    延迟映射 = 网络延迟.set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"].to_dict()
    功率系数 = 功率映射.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()
    任务 = 任务数据.copy()
    任务["任务序号"] = range(len(任务))
    任务["松弛时间"] = 任务["LatestFinishHour"] - np.maximum(任务["ArrivalHour"], 任务["EarliestStartHour"]) - 任务["EstimatedDuration_min"] / 60
    灵敏度序号 = {"High": 3, "Medium": 2, "Low": 1}
    任务["灵敏度序号"] = 任务["DelaySensitivity"].map(灵敏度序号).fillna(0)
    任务["GPU时长"] = 任务["GPU_Demand"] * 任务["EstimatedDuration_min"] / 60
    实时任务 = 任务[任务["TaskType"] == "RealTimeInference"].sort_values(["ArrivalHour", "GPU时长"], ascending=[True, False])
    弹性任务 = 任务[任务["TaskType"] != "RealTimeInference"].sort_values(
        ["松弛时间", "灵敏度序号", "GPU时长", "ArrivalHour"], ascending=[True, False, False, True]
    )
    排序任务 = pd.concat([实时任务, 弹性任务], ignore_index=True)
    调度记录 = []

    def 尝试安排(行):
        时长 = float(行["EstimatedDuration_min"]) / 60
        最早 = int(max(行["ArrivalHour"], 行["EarliestStartHour"]))
        最晚 = int(np.floor(min(行["LatestFinishHour"], 截止小时) - 时长 + 1e-10))
        开工列表 = [int(行["ArrivalHour"])] if 行["TaskType"] == "RealTimeInference" else range(最早, 最晚 + 1)
        候选区域 = []
        for 区域 in 区域列表:
            延迟 = float(延迟映射.get((行["SourceRegion"], 区域), np.inf))
            if 延迟 <= float(行["MaxLatency_ms"]):
                候选区域.append((区域, 延迟))
        for 开工 in 开工列表:
            完工 = 开工 + 时长
            if 开工 < 0 or 完工 > min(float(行["LatestFinishHour"]), 截止小时) + 1e-10:
                continue
            小时列表 = range(int(np.floor(开工)), int(np.ceil(完工 - 1e-12)))
            可行选择 = []
            for 区域, 延迟 in 候选区域:
                r = 区域序号[区域]
                最大利用率 = 0.0
                可行 = True
                for 小时 in 小时列表:
                    重叠 = 计算小时重叠(开工, 完工, 小时)
                    新GPU = GPU占用[r, 小时] + float(行["GPU_Demand"]) * 重叠
                    新AI功率 = AI功率[r, 小时] + float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 重叠
                    新IT功率 = 非AI功率[r, 小时] + 新AI功率
                    GPU利用率 = 新GPU / float(区域参数.loc[区域, "Available_GPU"])
                    IT利用率 = 新IT功率 / float(区域参数.loc[区域, "Max_IT_Power_MW"])
                    设施利用率 = 新IT功率 * float(区域参数.loc[区域, "PUE"]) / float(区域参数.loc[区域, "Max_Facility_Power_MW"])
                    if GPU利用率 > 1 + 1e-10 or IT利用率 > 1 + 1e-10 or 设施利用率 > 1 + 1e-10:
                        可行 = False
                        break
                    最大利用率 = max(最大利用率, GPU利用率, IT利用率)
                if 可行:
                    可行选择.append((最大利用率, 延迟, 区域))
            if 可行选择:
                _, 延迟, 区域 = min(可行选择)
                r = 区域序号[区域]
                for 小时 in 小时列表:
                    重叠 = 计算小时重叠(开工, 完工, 小时)
                    GPU占用[r, 小时] += float(行["GPU_Demand"]) * 重叠
                    AI功率[r, 小时] += float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 重叠
                return 区域, float(开工), float(完工), 延迟
        return None

    for _, 行 in 排序任务.iterrows():
        安排 = 尝试安排(行)
        if 安排 is None:
            调度记录.append([行["任务序号"], 行["TaskID"], "", np.nan, np.nan, np.nan, False])
        else:
            区域, 开工, 完工, 延迟 = 安排
            调度记录.append([行["任务序号"], 行["TaskID"], 区域, 开工, 完工, 延迟, True])
    调度结果 = pd.DataFrame(
        调度记录,
        columns=["任务序号", "TaskID", "ExecutionRegion", "StartHour", "FinishHour", "NetworkLatency_ms", "Scheduled"],
    ).sort_values("任务序号").drop(columns="任务序号").reset_index(drop=True)
    资源记录 = []
    for 区域 in 区域列表:
        r = 区域序号[区域]
        for 小时 in range(截止小时):
            总IT = 非AI功率[r, 小时] + AI功率[r, 小时]
            资源记录.append(
                [
                    小时,
                    区域,
                    GPU占用[r, 小时],
                    AI功率[r, 小时],
                    非AI功率[r, 小时],
                    总IT,
                    GPU占用[r, 小时] / float(区域参数.loc[区域, "Available_GPU"]),
                    总IT / float(区域参数.loc[区域, "Max_IT_Power_MW"]),
                    总IT * float(区域参数.loc[区域, "PUE"]) / float(区域参数.loc[区域, "Max_Facility_Power_MW"]),
                ]
            )
    资源结果 = pd.DataFrame(
        资源记录,
        columns=["Hour", "Region", "GPU_Used", "AI_IT_Power_MW", "NonAI_IT_Load_MW", "IT_Load_MW", "GPU_Utilization", "IT_Utilization", "Facility_Utilization"],
    )
    return 调度结果, 资源结果


def 检验调度约束(任务数据, 调度结果, 资源结果, 截止小时=2406):
    合并 = 任务数据.merge(调度结果, on="TaskID", how="left")
    已排程 = 合并["Scheduled"].fillna(False).astype(bool)
    检验值 = {
        "任务遗漏或未排程": int((~已排程).sum()),
        "任务重复排程": int(调度结果["TaskID"].duplicated().sum()),
        "实时任务未立即开工": int(
            (已排程 & (合并["TaskType"] == "RealTimeInference") & (np.abs(合并["StartHour"] - 合并["ArrivalHour"]) > 1e-9)).sum()
        ),
        "网络时延超限": int((已排程 & (合并["NetworkLatency_ms"] > 合并["MaxLatency_ms"] + 1e-9)).sum()),
        "任务截止时间或2406超限": int(
            (已排程 & ((合并["FinishHour"] > 合并["LatestFinishHour"] + 1e-9) | (合并["FinishHour"] > 截止小时 + 1e-9))).sum()
        ),
        "GPU容量超限": int((资源结果["GPU_Utilization"] > 1 + 1e-9).sum()),
        "IT功率超限": int((资源结果["IT_Utilization"] > 1 + 1e-9).sum()),
        "设施功率超限": int((资源结果["Facility_Utilization"] > 1 + 1e-9).sum()),
    }
    return pd.DataFrame(
        [{"检验项": 检验项, "违约数": 违约数, "是否通过": 违约数 == 0} for 检验项, 违约数 in 检验值.items()]
    )


def 执行双轨预测(需求序列):
    预测记录 = []
    指标记录 = []
    参数记录 = {}
    for 区域, 类型 in 需求序列.columns:
        全序列 = 需求序列[(区域, 类型)].reset_index(drop=True)
        训练序列 = 全序列.iloc[:2352]
        验证实际 = 全序列.iloc[2352:2376].reset_index(drop=True)
        测试实际 = 全序列.iloc[2376:2400].reset_index(drop=True)
        模型 = 选择预测模型(训练序列)
        零值率 = float((训练序列 == 0).mean())
        if 模型 == "ARIMA":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                验证结果 = ARIMA预测(训练序列, 24)
                最终结果 = ARIMA预测(全序列.iloc[:2376], 24, 指定阶数=验证结果["阶数"])
            验证预测 = 验证结果["预测值"]
            测试预测 = 最终结果["预测值"]
            参数 = {"model": 模型, "order": list(验证结果["阶数"]), "aic": 验证结果["AIC"]}
        else:
            最优系数 = None
            最优误差 = np.inf
            验证预测 = None
            for 系数 in np.arange(0.05, 0.31, 0.05):
                候选预测 = SBA克罗斯顿预测(训练序列, 24, float(系数))
                候选误差 = 计算预测指标(验证实际, 候选预测)["RMSE"]
                if 候选误差 < 最优误差:
                    最优误差 = 候选误差
                    最优系数 = float(系数)
                    验证预测 = 候选预测
            测试预测 = SBA克罗斯顿预测(全序列.iloc[:2376], 24, 最优系数)
            参数 = {"model": 模型, "alpha": 最优系数}
        验证指标 = 计算预测指标(验证实际, 验证预测)
        测试指标 = 计算预测指标(测试实际, 测试预测)
        参数记录[f"{区域}-{类型}"] = 参数
        指标记录.append(
            {
                "Region": 区域,
                "TaskType": 类型,
                "Model": 模型,
                "ZeroRate": 零值率,
                "Validation_RMSE": 验证指标["RMSE"],
                **测试指标,
            }
        )
        for 步数 in range(24):
            预测记录.append(
                {
                    "Hour": 2376 + 步数,
                    "Region": 区域,
                    "TaskType": 类型,
                    "Model": 模型,
                    "Actual_GPU_Demand": float(测试实际.iloc[步数]),
                    "Predicted_GPU_Demand": float(测试预测.iloc[步数]),
                }
            )
    return pd.DataFrame(预测记录), pd.DataFrame(指标记录), 参数记录


def 保存图形(图, 路径, 名称):
    图.tight_layout()
    图.savefig(路径 / f"{名称}.pdf", bbox_inches="tight")
    图.savefig(路径 / f"{名称}.png", dpi=220, bbox_inches="tight")
    plt.close(图)


def 配置绘图():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def 提取最后24小时结果(调度结果, 资源结果):
    末调度 = 调度结果[
        调度结果["ArrivalHour"].between(2376, 2399) & 调度结果["Scheduled"]
    ].copy()
    末资源 = 资源结果[资源结果["Hour"].between(2376, 2399)].copy()
    return 末调度, 末资源


def 绘制结果图(任务, 需求序列, 预测结果, 预测指标, 调度结果, 资源结果, 图形目录):
    配置绘图()
    蓝色 = ["#0B559F", "#2A7AB9", "#539DCC", "#88BEDC", "#BAD6EA"]
    红色 = ["#CE4459", "#E595A4", "#F7D4DB"]
    区域颜色 = 蓝色 + [红色[0]]
    类型顺序 = ["RealTimeInference", "BatchInference", "AITraining"]
    区域顺序 = sorted(任务["SourceRegion"].unique())

    图, 轴 = plt.subplots(1, 2, figsize=(12, 4.6))
    区域计数 = 任务["SourceRegion"].value_counts().reindex(区域顺序)
    类型计数 = 任务["TaskType"].value_counts().reindex(类型顺序)
    轴[0].bar(区域计数.index, 区域计数.values, color=区域颜色[: len(区域计数)])
    轴[0].set_title("各来源区域任务数量")
    轴[0].set_ylabel("任务数")
    轴[1].bar(类型计数.index, 类型计数.values, color=红色)
    轴[1].set_title("各任务类型数量")
    轴[1].tick_params(axis="x", rotation=15)
    保存图形(图, 图形目录, "图1_任务统计")

    小时需求 = 构造日内GPU总需求(需求序列)
    图, 轴 = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(小时需求.reindex(区域顺序), cmap=sns.light_palette("#2A7AB9", as_cmap=True), ax=轴)
    轴.set_title("区域日内平均每小时到达GPU总需求")
    轴.set_xlabel("Hour")
    轴.set_ylabel("区域")
    保存图形(图, 图形目录, "图2_时序需求热力图")

    零值表 = 预测指标.pivot(index="Region", columns="TaskType", values="ZeroRate").reindex(index=区域顺序, columns=类型顺序)
    标注 = 零值表.copy().astype(object)
    模型表 = 预测指标.pivot(index="Region", columns="TaskType", values="Model").reindex(index=区域顺序, columns=类型顺序)
    for 区域 in 区域顺序:
        for 类型 in 类型顺序:
            标注.loc[区域, 类型] = f"{零值表.loc[区域, 类型]:.1%}\n{模型表.loc[区域, 类型]}"
    图, 轴 = plt.subplots(figsize=(10, 5))
    sns.heatmap(零值表, annot=标注, fmt="", cmap=sns.light_palette("#CE4459", as_cmap=True), vmin=0, vmax=1, ax=轴)
    轴.set_title("训练段零值率与双轨模型分流")
    轴.set_xlabel("任务类型")
    轴.set_ylabel("区域")
    轴.tick_params(axis="y", rotation=0)
    保存图形(图, 图形目录, "图3_零值率与模型分轨")

    图, 轴 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for 序号, 模型 in enumerate(["ARIMA", "SBA-Croston"]):
        候选 = 预测指标[预测指标["Model"] == 模型].sort_values("RMSE")
        if len(候选) == 0:
            轴[序号].set_visible(False)
            continue
        行 = 候选.iloc[0]
        数据 = 预测结果[(预测结果["Region"] == 行["Region"]) & (预测结果["TaskType"] == 行["TaskType"])]
        轴[序号].plot(数据["Hour"], 数据["Actual_GPU_Demand"], color=蓝色[1], marker="o", label="实际值")
        轴[序号].plot(数据["Hour"], 数据["Predicted_GPU_Demand"], color=红色[0], marker="s", label="预测值")
        轴[序号].set_title(f"{模型}示例：{行['Region']}－{行['TaskType']}")
        轴[序号].set_ylabel("GPU需求")
        轴[序号].legend()
    轴[-1].set_xlabel("Hour")
    保存图形(图, 图形目录, "图4_双轨预测示例")

    误差表 = 预测指标.pivot(index="Region", columns="TaskType", values="WMAPE").reindex(index=区域顺序, columns=类型顺序)
    图, 轴 = plt.subplots(figsize=(9, 5))
    sns.heatmap(误差表, annot=True, fmt=".2f", cmap=sns.light_palette("#539DCC", as_cmap=True), ax=轴)
    轴.set_title("18条序列测试段WMAPE")
    轴.set_xlabel("任务类型")
    轴.set_ylabel("区域")
    轴.tick_params(axis="y", rotation=0)
    保存图形(图, 图形目录, "图5_预测误差热力图")

    末时段, 末资源 = 提取最后24小时结果(调度结果, 资源结果)
    图, 轴 = plt.subplots(figsize=(13, 6))
    类型颜色 = dict(zip(类型顺序, 红色))
    for 区域序号, 区域 in enumerate(区域顺序):
        当前区域 = 末时段[末时段["ExecutionRegion"] == 区域]
        for _, 行 in 当前区域.iterrows():
            轴.broken_barh([(行["StartHour"], 行["FinishHour"] - 行["StartHour"])], (区域序号 - 0.35, 0.7), facecolors=类型颜色[行["TaskType"]], alpha=0.55)
    轴.set_yticks(range(len(区域顺序)), 区域顺序)
    轴.set_xlim(2376, 2400)
    轴.set_xticks([2376, 2380, 2384, 2388, 2392, 2396, 2399])
    轴.set_xlabel("Hour")
    轴.set_ylabel("执行区域")
    轴.set_title("2376—2399时段任务调度甘特图")
    轴.legend(handles=[Patch(color=类型颜色[类型], label=类型) for 类型 in 类型顺序], loc="upper center", ncol=3)
    保存图形(图, 图形目录, "图6_末时段调度甘特图")

    图, 轴 = plt.subplots(figsize=(12, 5))
    for 序号, 区域 in enumerate(区域顺序):
        数据 = 末资源[末资源["Region"] == 区域]
        轴.plot(数据["Hour"], 数据["GPU_Utilization"] * 100, color=区域颜色[序号], marker="o", ms=3, label=区域)
    轴.axhline(100, color=红色[0], linestyle="--", linewidth=1, label="容量上限")
    轴.set_xlabel("Hour")
    轴.set_ylabel("GPU利用率 (%)")
    轴.set_xlim(2376, 2399)
    轴.set_xticks([2376, 2380, 2384, 2388, 2392, 2396, 2399])
    轴.set_title("最后24小时六区域GPU利用率")
    轴.legend(ncol=4)
    保存图形(图, 图形目录, "图7_区域GPU利用率")


def 生成结果报告(报告路径, 任务, 预测指标, 目标调度, 资源末段, 检验结果, 运行秒数):
    模型计数 = 预测指标["Model"].value_counts()
    成功数 = int(目标调度["Scheduled"].sum())
    目标数 = len(目标调度)
    平均指标 = 预测指标[["MAE", "RMSE", "WMAPE"]].mean()
    最大GPU = 资源末段.groupby("Region")["GPU_Utilization"].max()
    检验表格 = "| 检验项 | 违约数 | 是否通过 |\n|---|---:|:---:|\n" + "\n".join(
        f"| {行['检验项']} | {int(行['违约数'])} | {'是' if 行['是否通过'] else '否'} |" for _, 行 in 检验结果.iterrows()
    )
    文本 = f"""# 问题一计算结果

## 1 数据概况

共读取 {len(任务)} 条任务记录，来源区域数为 {任务['SourceRegion'].nunique()}，任务类型数为 {任务['TaskType'].nunique()}。预测对象为 18 条“来源区域—任务类型”小时GPU需求序列。

## 2 双轨预测结果

训练段为 0—2351 h，验证段为 2352—2375 h，测试段为 2376—2399 h。零值率低于 60% 的序列采用 ARIMA，达到 60% 的序列采用 SBA-Croston。本次分流得到 ARIMA {int(模型计数.get('ARIMA', 0))} 条，SBA-Croston {int(模型计数.get('SBA-Croston', 0))} 条。

18 条序列测试段平均 MAE 为 {平均指标['MAE']:.4f}，平均 RMSE 为 {平均指标['RMSE']:.4f}，平均 WMAPE 为 {平均指标['WMAPE']:.4f}。逐序列结果见 `预测指标.csv` 和 `预测结果.csv`。

## 3 基础调度结果

对 2376—2399 h 到达的 {目标数} 个任务进行输出，其中成功排程 {成功数} 个。实时推理任务固定在到达时刻开工；弹性任务按松弛时间、延迟敏感度和GPU时长排序，并在最早可行时刻选择分配后最大资源利用率最低的区域。

末时段各区域最大GPU利用率分别为：{', '.join(f'{区域} {数值:.2%}' for 区域, 数值 in 最大GPU.items())}。

## 4 必要检验

{检验表格}

仅保留预测误差和调度硬约束检验，不进行算法竞赛式对比，也不增加残差白噪声等扩展检验。

## 5 运行信息

总运行时间为 {运行秒数:.2f} s。全部数值由代码直接计算生成。
"""
    报告路径.write_text(文本, encoding="utf-8")


def 主程序():
    开始时间 = time.perf_counter()
    根目录 = Path(__file__).resolve().parents[1]
    数据目录 = 根目录 / "题目" / "附件数据"
    输出目录 = 根目录 / "outputs" / "问题一计算结果"
    图形目录 = 根目录 / "figures" / "问题一计算结果"
    报告目录 = 根目录 / "reports"
    输出目录.mkdir(parents=True, exist_ok=True)
    图形目录.mkdir(parents=True, exist_ok=True)
    报告目录.mkdir(parents=True, exist_ok=True)
    任务 = pd.read_excel(数据目录 / "workload_trace.xlsx")
    GPU信息 = pd.read_excel(数据目录 / "GPU_information.xlsx")
    时间数据 = pd.read_excel(数据目录 / "region_time_data.xlsx")
    网络延迟 = pd.read_excel(数据目录 / "network_latency.xlsx")
    功率映射 = pd.read_excel(数据目录 / "power_mapping.xlsx")
    区域列表 = sorted(任务["SourceRegion"].unique())
    类型列表 = ["RealTimeInference", "BatchInference", "AITraining"]
    需求序列 = 构造到达需求序列(任务, 2400, 区域列表, 类型列表)
    预测结果, 预测指标, 模型参数 = 执行双轨预测(需求序列)
    调度基础, 资源结果 = 执行基础调度(任务, GPU信息, 时间数据, 网络延迟, 功率映射, 2406)
    完整调度 = 任务.merge(调度基础, on="TaskID", how="left")
    检验结果 = 检验调度约束(任务, 调度基础, 资源结果, 2406)
    目标调度 = 完整调度[(完整调度["ArrivalHour"] >= 2376) & (完整调度["ArrivalHour"] <= 2399)].copy()
    资源末段 = 资源结果[(资源结果["Hour"] >= 2376) & (资源结果["Hour"] <= 2405)].copy()
    数据统计 = 任务.groupby(["SourceRegion", "TaskType"], as_index=False).agg(
        Task_Count=("TaskID", "count"),
        GPU_Demand_Total=("GPU_Demand", "sum"),
        Duration_min_Mean=("EstimatedDuration_min", "mean"),
    )
    零值率 = 预测指标[["Region", "TaskType", "ZeroRate"]].rename(columns={"Region": "SourceRegion"})
    数据统计 = 数据统计.merge(零值率, on=["SourceRegion", "TaskType"], how="left")
    数据统计.to_csv(输出目录 / "数据统计.csv", index=False, encoding="utf-8-sig")
    预测结果.to_csv(输出目录 / "预测结果.csv", index=False, encoding="utf-8-sig")
    预测指标.to_csv(输出目录 / "预测指标.csv", index=False, encoding="utf-8-sig")
    图2数据 = 构造日内GPU总需求(需求序列).rename_axis("Region").reset_index().melt(
        id_vars="Region", var_name="HourOfDay", value_name="Mean_Arrival_GPU_Demand"
    )
    图2数据.to_csv(输出目录 / "区域日内小时GPU总需求.csv", index=False, encoding="utf-8-sig")
    目标调度.to_csv(输出目录 / "逐任务调度.csv", index=False, encoding="utf-8-sig")
    资源末段.to_csv(输出目录 / "区域逐时资源.csv", index=False, encoding="utf-8-sig")
    检验结果.to_csv(输出目录 / "约束检验.csv", index=False, encoding="utf-8-sig")
    (输出目录 / "模型参数.json").write_text(json.dumps(模型参数, ensure_ascii=False, indent=2), encoding="utf-8")
    绘制结果图(任务, 需求序列, 预测结果, 预测指标, 完整调度, 资源结果, 图形目录)
    运行秒数 = time.perf_counter() - 开始时间
    运行统计 = {
        "任务总数": len(任务),
        "目标时段任务数": len(目标调度),
        "目标时段成功排程数": int(目标调度["Scheduled"].sum()),
        "预测序列数": len(预测指标),
        "运行时间_s": 运行秒数,
    }
    (输出目录 / "运行统计.json").write_text(json.dumps(运行统计, ensure_ascii=False, indent=2), encoding="utf-8")
    生成结果报告(报告目录 / "问题一计算结果.md", 任务, 预测指标, 目标调度, 资源末段, 检验结果, 运行秒数)
    print(json.dumps(运行统计, ensure_ascii=False, indent=2))
    print(检验结果.to_string(index=False))


if __name__ == "__main__":
    主程序()
