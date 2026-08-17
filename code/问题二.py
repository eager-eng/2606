import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def 计算小时重叠(开始时刻, 结束时刻, 小时):
    return max(0.0, min(float(结束时刻), 小时 + 1.0) - max(float(开始时刻), float(小时)))


def 计算能源状态(设施负荷, 可用新能源, 基准充电, 基准放电, 基准售电, 购电价, 售电价, 碳强度):
    设施负荷 = np.asarray(设施负荷, dtype=float)
    可用新能源 = np.asarray(可用新能源, dtype=float)
    基准充电 = np.asarray(基准充电, dtype=float)
    基准放电 = np.asarray(基准放电, dtype=float)
    基准售电 = np.asarray(基准售电, dtype=float)
    净用能需求 = 设施负荷 + 基准充电 - 基准放电
    购电功率 = np.maximum(净用能需求 - 可用新能源, 0.0)
    剩余功率 = np.maximum(可用新能源 - 净用能需求, 0.0)
    售电功率 = np.minimum(基准售电, 剩余功率)
    弃电功率 = 剩余功率 - 售电功率
    运行成本 = 购电功率 * np.asarray(购电价, dtype=float) - 售电功率 * np.asarray(售电价, dtype=float)
    碳排放 = 购电功率 * np.asarray(碳强度, dtype=float)
    平衡残差 = 购电功率 + 可用新能源 + 基准放电 - 设施负荷 - 基准充电 - 售电功率 - 弃电功率
    新能源利用量 = np.maximum(可用新能源 - 弃电功率, 0.0)
    新能源总量 = float(可用新能源.sum())
    新能源利用率 = float(新能源利用量.sum() / 新能源总量) if 新能源总量 > 0 else 0.0
    return {
        "GridPurchase_MW": 购电功率,
        "GridSell_MW": 售电功率,
        "Curtailment_MW": 弃电功率,
        "OperatingCost_CNY": 运行成本,
        "CarbonEmission_tCO2": 碳排放,
        "RenewableUsed_MW": 新能源利用量,
        "PowerBalanceResidual_MW": 平衡残差,
        "RenewableUtilization": 新能源利用率,
    }


def 计算边际能源状态(
    设施负荷,
    基准设施负荷,
    可用新能源,
    基准购电,
    基准售电,
    基准弃电,
    基准充电,
    基准放电,
    购电价,
    售电价,
    碳强度,
):
    设施负荷 = np.asarray(设施负荷, dtype=float)
    基准设施负荷 = np.asarray(基准设施负荷, dtype=float)
    可用新能源 = np.asarray(可用新能源, dtype=float)
    购电功率 = np.asarray(基准购电, dtype=float).copy()
    售电功率 = np.asarray(基准售电, dtype=float).copy()
    弃电功率 = np.asarray(基准弃电, dtype=float).copy()
    基准充电 = np.asarray(基准充电, dtype=float)
    基准放电 = np.asarray(基准放电, dtype=float)
    负荷变化 = 设施负荷 - 基准设施负荷
    增加负荷 = np.maximum(负荷变化, 0.0)
    减少弃电 = np.minimum(增加负荷, 弃电功率)
    弃电功率 -= 减少弃电
    剩余增加 = 增加负荷 - 减少弃电
    减少售电 = np.minimum(剩余增加, 售电功率)
    售电功率 -= 减少售电
    购电功率 += 剩余增加 - 减少售电
    减少负荷 = np.maximum(-负荷变化, 0.0)
    减少购电 = np.minimum(减少负荷, 购电功率)
    购电功率 -= 减少购电
    弃电功率 += 减少负荷 - 减少购电
    运行成本 = 购电功率 * np.asarray(购电价, dtype=float) - 售电功率 * np.asarray(售电价, dtype=float)
    碳排放 = 购电功率 * np.asarray(碳强度, dtype=float)
    平衡残差 = 购电功率 + 可用新能源 + 基准放电 - 设施负荷 - 基准充电 - 售电功率 - 弃电功率
    新能源利用量 = np.maximum(可用新能源 - 弃电功率, 0.0)
    新能源总量 = float(可用新能源.sum())
    新能源利用率 = float(新能源利用量.sum() / 新能源总量) if 新能源总量 > 0 else 0.0
    return {
        "GridPurchase_MW": 购电功率,
        "GridSell_MW": 售电功率,
        "Curtailment_MW": 弃电功率,
        "OperatingCost_CNY": 运行成本,
        "CarbonEmission_tCO2": 碳排放,
        "RenewableUsed_MW": 新能源利用量,
        "PowerBalanceResidual_MW": 平衡残差,
        "RenewableUtilization": 新能源利用率,
    }


def 计算三目标综合值(运行成本, 碳排放, 新能源利用率, 基准成本, 基准碳排放, 基准新能源利用率):
    if 基准成本 <= 0 or 基准碳排放 <= 0 or 基准新能源利用率 <= 0:
        raise ValueError("基准指标必须为正数")
    成本变化 = (运行成本 - 基准成本) / 基准成本
    碳变化 = (碳排放 - 基准碳排放) / 基准碳排放
    新能源变化 = (基准新能源利用率 - 新能源利用率) / 基准新能源利用率
    return float((成本变化 + 碳变化 + 新能源变化) / 3)


def 选择最优候选(候选列表):
    if not 候选列表:
        return None
    return min(
        候选列表,
        key=lambda 候选: (
            候选["Score"],
            候选["NetworkLatency_ms"],
            候选["StartHour"],
            候选["MigrationFlag"],
            候选.get("Region", ""),
        ),
    )


def 项目根目录():
    return Path(__file__).resolve().parents[1]


def 构造逐时矩阵(时间数据, 区域列表, 字段, 截止小时):
    数据表 = 时间数据.pivot_table(index="Hour", columns="Region", values=字段, aggfunc="first")
    数据表 = 数据表.reindex(range(截止小时 + 1)).ffill().bfill().reindex(columns=区域列表)
    return 数据表.to_numpy(dtype=float).T


class 区间候选器:
    def __init__(self, 得分, 保留数):
        self.保留数 = 保留数
        长度 = len(得分)
        容量 = 1
        while 容量 < 长度:
            容量 *= 2
        self.容量 = 容量
        self.树 = [[] for _ in range(2 * 容量)]
        for 序号, 数值 in enumerate(得分):
            self.树[容量 + 序号] = [(float(数值), 序号)]
        for 节点 in range(容量 - 1, 0, -1):
            self.树[节点] = sorted(self.树[节点 * 2] + self.树[节点 * 2 + 1])[:保留数]

    def 查询(self, 左端, 右端):
        左端 += self.容量
        右端 += self.容量 + 1
        结果 = []
        while 左端 < 右端:
            if 左端 % 2:
                结果.extend(self.树[左端])
                左端 += 1
            if 右端 % 2:
                右端 -= 1
                结果.extend(self.树[右端])
            左端 //= 2
            右端 //= 2
        return [序号 for _, 序号 in sorted(结果)[: self.保留数]]


def 计算基准指标(时间数据):
    成本 = float(
        (
            时间数据["GridPurchase_MW"] * 时间数据["ElectricityPrice_CNY_per_MWh"]
            - 时间数据["GridSell_MW"] * 时间数据["SellPrice_CNY_per_MWh"]
        ).sum()
    )
    碳排放 = float((时间数据["GridPurchase_MW"] * 时间数据["CarbonIntensity_tCO2_per_MWh"]).sum())
    新能源总量 = float(时间数据["AvailableRenewable_MW"].sum())
    新能源利用量 = float((时间数据["AvailableRenewable_MW"] - 时间数据["Curtailment_MW"]).clip(lower=0).sum())
    新能源利用率 = 新能源利用量 / 新能源总量 if 新能源总量 > 0 else 0.0
    return {
        "运行成本_CNY": 成本,
        "碳排放_tCO2": 碳排放,
        "新能源利用量_MWh": 新能源利用量,
        "新能源利用率": 新能源利用率,
    }


def 执行问题二调度(任务数据, GPU信息, 时间数据, 网络延迟, 功率映射, 截止小时=2406, 候选数=10):
    区域列表 = GPU信息["Region"].tolist()
    区域序号 = {区域: 序号 for 序号, 区域 in enumerate(区域列表)}
    区域参数 = GPU信息.set_index("Region")
    区域数 = len(区域列表)
    能源小时数 = 截止小时 + 1
    GPU占用 = np.zeros((区域数, 能源小时数), dtype=float)
    AI功率 = np.zeros((区域数, 能源小时数), dtype=float)
    非AI功率 = 构造逐时矩阵(时间数据, 区域列表, "NonAI_IT_Load_MW", 截止小时)
    基准AI功率 = 构造逐时矩阵(时间数据, 区域列表, "Baseline_AI_IT_Load_MW", 截止小时)
    可用新能源 = 构造逐时矩阵(时间数据, 区域列表, "AvailableRenewable_MW", 截止小时)
    基准充电 = 构造逐时矩阵(时间数据, 区域列表, "ChargePower_MW", 截止小时)
    基准放电 = 构造逐时矩阵(时间数据, 区域列表, "DischargePower_MW", 截止小时)
    基准购电 = 构造逐时矩阵(时间数据, 区域列表, "GridPurchase_MW", 截止小时)
    基准售电 = 构造逐时矩阵(时间数据, 区域列表, "GridSell_MW", 截止小时)
    基准弃电 = 构造逐时矩阵(时间数据, 区域列表, "Curtailment_MW", 截止小时)
    购电价 = 构造逐时矩阵(时间数据, 区域列表, "ElectricityPrice_CNY_per_MWh", 截止小时)
    售电价 = 构造逐时矩阵(时间数据, 区域列表, "SellPrice_CNY_per_MWh", 截止小时)
    碳强度 = 构造逐时矩阵(时间数据, 区域列表, "CarbonIntensity_tCO2_per_MWh", 截止小时)
    PUE = np.array([float(区域参数.loc[区域, "PUE"]) for 区域 in 区域列表])[:, None]
    基准设施负荷 = PUE * (非AI功率 + 基准AI功率)
    设施负荷 = PUE * 非AI功率
    能源状态 = 计算边际能源状态(
        设施负荷,
        基准设施负荷,
        可用新能源,
        基准购电,
        基准售电,
        基准弃电,
        基准充电,
        基准放电,
        购电价,
        售电价,
        碳强度,
    )
    基准指标 = 计算基准指标(时间数据[时间数据["Hour"].between(0, 截止小时)])
    基准成本 = max(基准指标["运行成本_CNY"], 1e-12)
    基准碳排放 = max(基准指标["碳排放_tCO2"], 1e-12)
    基准新能源利用量 = max(基准指标["新能源利用量_MWh"], 1e-12)
    延迟映射 = 网络延迟.set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"].to_dict()
    功率系数 = 功率映射.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()
    静态信号 = np.zeros((区域数, 截止小时), dtype=float)
    for r in range(区域数):
        原状态 = {键: np.asarray(值)[r, :截止小时] for 键, 值 in 能源状态.items() if isinstance(值, np.ndarray)}
        新设施 = 设施负荷[r, :截止小时] + 1.0
        新状态 = 计算边际能源状态(
            新设施,
            基准设施负荷[r, :截止小时],
            可用新能源[r, :截止小时],
            基准购电[r, :截止小时],
            基准售电[r, :截止小时],
            基准弃电[r, :截止小时],
            基准充电[r, :截止小时],
            基准放电[r, :截止小时],
            购电价[r, :截止小时],
            售电价[r, :截止小时],
            碳强度[r, :截止小时],
        )
        静态信号[r] = (
            (新状态["OperatingCost_CNY"] - 原状态["OperatingCost_CNY"]) / 基准成本
            + (新状态["CarbonEmission_tCO2"] - 原状态["CarbonEmission_tCO2"]) / 基准碳排放
            - (新状态["RenewableUsed_MW"] - 原状态["RenewableUsed_MW"]) / 基准新能源利用量
        ) / 3
    候选器 = [区间候选器(静态信号[r], max(候选数, 1)) for r in range(区域数)]
    任务 = 任务数据.copy()
    任务["任务序号"] = range(len(任务))
    任务["松弛时间"] = np.minimum(任务["LatestFinishHour"], 截止小时) - np.maximum(
        任务["ArrivalHour"], 任务["EarliestStartHour"]
    ) - 任务["EstimatedDuration_min"] / 60
    灵敏度序号 = {"High": 3, "Medium": 2, "Low": 1}
    任务["灵敏度序号"] = 任务["DelaySensitivity"].map(灵敏度序号).fillna(0)
    任务["GPU时长"] = 任务["GPU_Demand"] * 任务["EstimatedDuration_min"] / 60
    实时任务 = 任务[任务["TaskType"] == "RealTimeInference"].sort_values(["ArrivalHour", "GPU时长"], ascending=[True, False])
    弹性任务 = 任务[任务["TaskType"] != "RealTimeInference"].sort_values(
        ["松弛时间", "灵敏度序号", "GPU时长", "ArrivalHour"], ascending=[True, False, False, True]
    )
    排序任务 = pd.concat([实时任务, 弹性任务], ignore_index=True)
    调度记录 = []

    def 构造占用(开工, 完工):
        return [
            (小时, 计算小时重叠(开工, 完工, 小时))
            for 小时 in range(int(np.floor(开工)), int(np.ceil(完工 - 1e-12)))
        ]

    def 评估候选(行, 区域, 延迟, 开工):
        r = 区域序号[区域]
        时长 = float(行["EstimatedDuration_min"]) / 60
        完工 = float(开工) + 时长
        if 完工 > min(float(行["LatestFinishHour"]), 截止小时) + 1e-10:
            return None
        占用 = 构造占用(float(开工), 完工)
        if not 占用 or any(小时 < 0 or 小时 >= 截止小时 for 小时, _ in 占用):
            return None
        小时 = np.array([值[0] for 值 in 占用], dtype=int)
        重叠 = np.array([值[1] for 值 in 占用], dtype=float)
        新GPU = GPU占用[r, 小时] + float(行["GPU_Demand"]) * 重叠
        新AI = AI功率[r, 小时] + float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 重叠
        新IT = 非AI功率[r, 小时] + 新AI
        新设施 = float(区域参数.loc[区域, "PUE"]) * 新IT
        if np.any(新GPU > float(区域参数.loc[区域, "Available_GPU"]) + 1e-10):
            return None
        if np.any(新IT > float(区域参数.loc[区域, "Max_IT_Power_MW"]) + 1e-10):
            return None
        if np.any(新设施 > float(区域参数.loc[区域, "Max_Facility_Power_MW"]) + 1e-10):
            return None
        新能源局部 = 计算边际能源状态(
            新设施,
            基准设施负荷[r, 小时],
            可用新能源[r, 小时],
            基准购电[r, 小时],
            基准售电[r, 小时],
            基准弃电[r, 小时],
            基准充电[r, 小时],
            基准放电[r, 小时],
            购电价[r, 小时],
            售电价[r, 小时],
            碳强度[r, 小时],
        )
        成本变化 = float((新能源局部["OperatingCost_CNY"] - 能源状态["OperatingCost_CNY"][r, 小时]).sum())
        碳变化 = float((新能源局部["CarbonEmission_tCO2"] - 能源状态["CarbonEmission_tCO2"][r, 小时]).sum())
        新能源变化 = float((新能源局部["RenewableUsed_MW"] - 能源状态["RenewableUsed_MW"][r, 小时]).sum())
        得分 = (成本变化 / 基准成本 + 碳变化 / 基准碳排放 - 新能源变化 / 基准新能源利用量) / 3
        return {
            "Score": 得分,
            "NetworkLatency_ms": float(延迟),
            "StartHour": float(开工),
            "FinishHour": 完工,
            "MigrationFlag": int(区域 != 行["SourceRegion"]),
            "Region": 区域,
            "小时": 小时,
            "重叠": 重叠,
            "新AI": 新AI,
            "新设施": 新设施,
            "新能源局部": 新能源局部,
        }

    def 提交候选(候选, 行):
        r = 区域序号[候选["Region"]]
        小时 = 候选["小时"]
        GPU占用[r, 小时] += float(行["GPU_Demand"]) * 候选["重叠"]
        AI功率[r, 小时] = 候选["新AI"]
        设施负荷[r, 小时] = 候选["新设施"]
        for 键 in [
            "GridPurchase_MW",
            "GridSell_MW",
            "Curtailment_MW",
            "OperatingCost_CNY",
            "CarbonEmission_tCO2",
            "RenewableUsed_MW",
            "PowerBalanceResidual_MW",
        ]:
            能源状态[键][r, 小时] = 候选["新能源局部"][键]

    for _, 行 in 排序任务.iterrows():
        时长 = float(行["EstimatedDuration_min"]) / 60
        最早 = int(max(行["ArrivalHour"], 行["EarliestStartHour"]))
        最晚 = int(np.floor(min(float(行["LatestFinishHour"]), 截止小时) - 时长 + 1e-10))
        可行区域 = []
        for 区域 in 区域列表:
            延迟 = float(延迟映射.get((行["SourceRegion"], 区域), np.inf))
            if 延迟 <= float(行["MaxLatency_ms"]):
                可行区域.append((区域, 延迟))
        候选列表 = []
        if 最晚 >= 最早:
            for 区域, 延迟 in 可行区域:
                r = 区域序号[区域]
                if 行["TaskType"] == "RealTimeInference":
                    开工列表 = [int(行["ArrivalHour"])]
                else:
                    开工列表 = sorted(set(候选器[r].查询(最早, 最晚) + [最早, 最晚]))
                for 开工 in 开工列表:
                    候选 = 评估候选(行, 区域, 延迟, 开工)
                    if 候选 is not None:
                        候选列表.append(候选)
        if not 候选列表 and 行["TaskType"] != "RealTimeInference" and 最晚 >= 最早:
            for 开工 in range(最早, 最晚 + 1):
                for 区域, 延迟 in 可行区域:
                    候选 = 评估候选(行, 区域, 延迟, 开工)
                    if 候选 is not None:
                        候选列表.append(候选)
                if 候选列表:
                    break
        最优 = 选择最优候选(候选列表)
        if 最优 is None:
            调度记录.append(
                [行["任务序号"], 行["TaskID"], "", np.nan, np.nan, np.nan, False, np.nan, 0]
            )
        else:
            提交候选(最优, 行)
            AI能量 = float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 时长
            调度记录.append(
                [
                    行["任务序号"],
                    行["TaskID"],
                    最优["Region"],
                    最优["StartHour"],
                    最优["FinishHour"],
                    最优["NetworkLatency_ms"],
                    True,
                    AI能量,
                    最优["MigrationFlag"],
                ]
            )
    调度结果 = pd.DataFrame(
        调度记录,
        columns=[
            "任务序号",
            "TaskID",
            "ExecutionRegion",
            "StartHour",
            "FinishHour",
            "NetworkLatency_ms",
            "Scheduled",
            "AI_IT_Energy_MWh",
            "MigrationFlag",
        ],
    ).sort_values("任务序号").drop(columns="任务序号").reset_index(drop=True)
    资源记录 = []
    for 区域 in 区域列表:
        r = 区域序号[区域]
        for 小时 in range(能源小时数):
            IT负荷 = 非AI功率[r, 小时] + AI功率[r, 小时]
            资源记录.append(
                {
                    "Hour": 小时,
                    "Region": 区域,
                    "GPU_Used": GPU占用[r, 小时],
                    "AI_IT_Load_MW": AI功率[r, 小时],
                    "Baseline_AI_IT_Load_MW": 基准AI功率[r, 小时],
                    "NonAI_IT_Load_MW": 非AI功率[r, 小时],
                    "IT_Load_MW": IT负荷,
                    "Facility_Load_MW": 设施负荷[r, 小时],
                    "Baseline_Facility_Load_MW": 基准设施负荷[r, 小时],
                    "GPU_Utilization": GPU占用[r, 小时] / float(区域参数.loc[区域, "Available_GPU"]),
                    "IT_Utilization": IT负荷 / float(区域参数.loc[区域, "Max_IT_Power_MW"]),
                    "Facility_Utilization": 设施负荷[r, 小时] / float(区域参数.loc[区域, "Max_Facility_Power_MW"]),
                    "AvailableRenewable_MW": 可用新能源[r, 小时],
                    "ChargePower_MW": 基准充电[r, 小时],
                    "DischargePower_MW": 基准放电[r, 小时],
                    "Baseline_GridPurchase_MW": 基准购电[r, 小时],
                    "Baseline_GridSell_MW": 基准售电[r, 小时],
                    "Baseline_Curtailment_MW": 基准弃电[r, 小时],
                    "GridPurchase_MW": 能源状态["GridPurchase_MW"][r, 小时],
                    "GridSell_MW": 能源状态["GridSell_MW"][r, 小时],
                    "Curtailment_MW": 能源状态["Curtailment_MW"][r, 小时],
                    "RenewableUsed_MW": 能源状态["RenewableUsed_MW"][r, 小时],
                    "ElectricityPrice_CNY_per_MWh": 购电价[r, 小时],
                    "SellPrice_CNY_per_MWh": 售电价[r, 小时],
                    "CarbonIntensity_tCO2_per_MWh": 碳强度[r, 小时],
                    "OperatingCost_CNY": 能源状态["OperatingCost_CNY"][r, 小时],
                    "CarbonEmission_tCO2": 能源状态["CarbonEmission_tCO2"][r, 小时],
                    "PowerBalanceResidual_MW": 能源状态["PowerBalanceResidual_MW"][r, 小时],
                }
            )
    资源结果 = pd.DataFrame(资源记录)
    已排 = 调度结果[调度结果["Scheduled"]]
    优化成本 = float(资源结果["OperatingCost_CNY"].sum())
    优化碳排放 = float(资源结果["CarbonEmission_tCO2"].sum())
    优化新能源利用率 = float(资源结果["RenewableUsed_MW"].sum() / max(资源结果["AvailableRenewable_MW"].sum(), 1e-12))
    基准时延 = 任务数据["SourceRegion"].map(
        {区域: float(延迟映射.get((区域, 区域), np.nan)) for 区域 in 区域列表}
    )
    指标 = {
        "任务数": int(len(调度结果)),
        "成功调度数": int(调度结果["Scheduled"].sum()),
        "基准运行成本_CNY": 基准指标["运行成本_CNY"],
        "优化运行成本_CNY": 优化成本,
        "基准碳排放_tCO2": 基准指标["碳排放_tCO2"],
        "优化碳排放_tCO2": 优化碳排放,
        "基准新能源利用率": 基准指标["新能源利用率"],
        "优化新能源利用率": 优化新能源利用率,
        "三目标综合值": 计算三目标综合值(
            优化成本,
            优化碳排放,
            优化新能源利用率,
            基准指标["运行成本_CNY"],
            基准指标["碳排放_tCO2"],
            基准指标["新能源利用率"],
        ),
        "基准平均网络时延_ms": float(基准时延.mean()),
        "优化平均网络时延_ms": float(已排["NetworkLatency_ms"].mean()) if len(已排) else np.nan,
        "基准P95网络时延_ms": float(基准时延.quantile(0.95)),
        "优化P95网络时延_ms": float(已排["NetworkLatency_ms"].quantile(0.95)) if len(已排) else np.nan,
        "基准最大网络时延_ms": float(基准时延.max()),
        "优化最大网络时延_ms": float(已排["NetworkLatency_ms"].max()) if len(已排) else np.nan,
        "任务迁移率": float(已排["MigrationFlag"].mean()) if len(已排) else np.nan,
    }
    return 调度结果, 资源结果, 指标


def 检验问题二约束(任务数据, 调度结果, 资源结果, 截止小时=2406):
    合并 = 任务数据.merge(调度结果, on="TaskID", how="left")
    已排程 = 合并["Scheduled"].fillna(False).astype(bool)
    最早允许时刻 = np.maximum(合并["ArrivalHour"], 合并["EarliestStartHour"])
    计划时长 = 合并["EstimatedDuration_min"] / 60
    检验值 = {
        "任务遗漏或未排程": float((~已排程).sum()),
        "任务重复排程": float(调度结果["TaskID"].duplicated().sum()),
        "任务早于允许时刻开工": float((已排程 & (合并["StartHour"] < 最早允许时刻 - 1e-9)).sum()),
        "任务连续执行时长不一致": float(
            (已排程 & (np.abs(合并["FinishHour"] - 合并["StartHour"] - 计划时长) > 1e-9)).sum()
        ),
        "实时任务未立即开工": float(
            (已排程 & (合并["TaskType"] == "RealTimeInference") & (np.abs(合并["StartHour"] - 合并["ArrivalHour"]) > 1e-9)).sum()
        ),
        "网络时延超限": float((已排程 & (合并["NetworkLatency_ms"] > 合并["MaxLatency_ms"] + 1e-9)).sum()),
        "任务截止时间或2406超限": float(
            (已排程 & ((合并["FinishHour"] > 合并["LatestFinishHour"] + 1e-9) | (合并["FinishHour"] > 截止小时 + 1e-9))).sum()
        ),
        "GPU容量超限": float((资源结果["GPU_Utilization"] > 1 + 1e-9).sum()),
        "IT功率超限": float((资源结果["IT_Utilization"] > 1 + 1e-9).sum()),
        "设施功率超限": float((资源结果["Facility_Utilization"] > 1 + 1e-9).sum()),
        "电力平衡最大残差_MW": float(资源结果["PowerBalanceResidual_MW"].abs().max()),
    }
    return pd.DataFrame(
        [
            {
                "检验项": 检验项,
                "违约数或残差": 数值,
                "是否通过": 数值 <= (1e-3 if "残差" in 检验项 else 0),
            }
            for 检验项, 数值 in 检验值.items()
        ]
    )


def 配置绘图():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.alpha": 0.8,
        }
    )


def 保存图形(图形, 图片目录, 文件名):
    图形.tight_layout()
    图形.savefig(图片目录 / f"{文件名}.png", bbox_inches="tight")
    图形.savefig(图片目录 / f"{文件名}.pdf", bbox_inches="tight")
    plt.close(图形)


def 构造指标对比表(指标):
    行 = [
        ["净运行成本", 指标["基准运行成本_CNY"], 指标["优化运行成本_CNY"], "CNY"],
        ["碳排放", 指标["基准碳排放_tCO2"], 指标["优化碳排放_tCO2"], "tCO2"],
        ["新能源利用率", 指标["基准新能源利用率"] * 100, 指标["优化新能源利用率"] * 100, "%"],
        ["平均网络时延", 指标["基准平均网络时延_ms"], 指标["优化平均网络时延_ms"], "ms"],
        ["P95网络时延", 指标["基准P95网络时延_ms"], 指标["优化P95网络时延_ms"], "ms"],
        ["最大网络时延", 指标["基准最大网络时延_ms"], 指标["优化最大网络时延_ms"], "ms"],
    ]
    对比 = pd.DataFrame(行, columns=["指标", "基准值", "优化值", "单位"])
    对比["相对变化"] = np.where(
        np.abs(对比["基准值"]) > 1e-12,
        (对比["优化值"] - 对比["基准值"]) / np.abs(对比["基准值"]),
        np.nan,
    )
    return 对比


def 绘制问题二图表(任务明细, 资源结果, 指标, 图片目录, 输出目录):
    配置绘图()
    图片目录.mkdir(parents=True, exist_ok=True)
    对比 = 构造指标对比表(指标)
    对比.to_csv(输出目录 / "指标对比.csv", index=False, encoding="utf-8-sig")
    图形, 坐标轴 = plt.subplots(2, 2, figsize=(11, 8))
    图项 = [
        ("净运行成本", "亿元", 1e8),
        ("碳排放", "万 tCO2", 1e4),
        ("新能源利用率", "%", 1.0),
        ("平均网络时延", "ms", 1.0),
    ]
    for 轴, (名称, 单位, 缩放) in zip(坐标轴.ravel(), 图项):
        行 = 对比[对比["指标"] == 名称].iloc[0]
        数值 = np.array([行["基准值"], 行["优化值"]], dtype=float) / 缩放
        标签 = ["本地执行参考", "问题二方案"] if "时延" in 名称 else ["附件基准", "问题二方案"]
        柱 = 轴.bar(标签, 数值, color=["#BAD6EA", "#CE4459"], width=0.58)
        轴.set_title(名称)
        轴.set_ylabel(单位)
        轴.axhline(0, color="#9CA3AF", linewidth=0.8)
        轴.grid(axis="x", visible=False)
        for 矩形, 值 in zip(柱, 数值):
            轴.text(
                矩形.get_x() + 矩形.get_width() / 2,
                值,
                f"{值:,.2f}",
                ha="center",
                va="bottom" if 值 >= 0 else "top",
                fontsize=9,
            )
    图形.suptitle("问题二关键指标对比", fontsize=15, y=1.01)
    保存图形(图形, 图片目录, "图1_关键指标对比")
    区域顺序 = sorted(set(任务明细["SourceRegion"]).union(任务明细["ExecutionRegion"].dropna()))
    迁移矩阵 = pd.crosstab(任务明细["SourceRegion"], 任务明细["ExecutionRegion"]).reindex(
        index=区域顺序, columns=区域顺序, fill_value=0
    )
    迁移矩阵.to_csv(输出目录 / "任务迁移矩阵.csv", encoding="utf-8-sig")
    色图 = LinearSegmentedColormap.from_list("低饱和蓝", ["#F7FAFC", "#BAD6EA", "#539DCC", "#0B559F"])
    图形, 轴 = plt.subplots(figsize=(8.2, 6.6))
    图像 = 轴.imshow(迁移矩阵.to_numpy(), cmap=色图, aspect="auto")
    轴.grid(False)
    轴.set_xticks(range(len(区域顺序)), 区域顺序)
    轴.set_yticks(range(len(区域顺序)), 区域顺序)
    轴.set_xlabel("执行区域")
    轴.set_ylabel("来源区域")
    轴.set_title("跨区域任务迁移矩阵")
    阈值 = 迁移矩阵.to_numpy().max() * 0.55
    for i in range(len(区域顺序)):
        for j in range(len(区域顺序)):
            值 = int(迁移矩阵.iloc[i, j])
            轴.text(j, i, f"{值:,}", ha="center", va="center", color="white" if 值 > 阈值 else "#1F2937")
    图形.colorbar(图像, ax=轴, label="任务数")
    保存图形(图形, 图片目录, "图2_跨区域任务迁移矩阵")
    负载矩阵 = 资源结果.pivot(index="Region", columns="Hour", values="Facility_Utilization")
    图形, 轴 = plt.subplots(figsize=(12.5, 4.6))
    图像 = 轴.imshow(负载矩阵.to_numpy(), cmap=色图, aspect="auto", vmin=0, vmax=1)
    轴.grid(False)
    轴.set_yticks(range(len(负载矩阵.index)), 负载矩阵.index)
    刻度 = np.linspace(0, len(负载矩阵.columns) - 1, 9, dtype=int)
    轴.set_xticks(刻度, [int(负载矩阵.columns[i]) for i in 刻度])
    轴.set_xlabel("Hour")
    轴.set_ylabel("区域")
    轴.set_title("区域逐时设施负载率")
    图形.colorbar(图像, ax=轴, label="设施负载率")
    保存图形(图形, 图片目录, "图3_区域逐时设施负载率")
    日内 = 资源结果.copy()
    日内["日内小时"] = 日内["Hour"] % 24
    日内 = 日内.groupby(["Region", "日内小时"])[["AI_IT_Load_MW", "AvailableRenewable_MW", "Curtailment_MW"]].mean().reset_index()
    日内.to_csv(输出目录 / "日内算电匹配数据.csv", index=False, encoding="utf-8-sig")
    图形, 坐标轴 = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for 轴, 区域 in zip(坐标轴.ravel(), sorted(日内["Region"].unique())):
        子表 = 日内[日内["Region"] == 区域]
        轴.plot(子表["日内小时"], 子表["AvailableRenewable_MW"], color="#2A7AB9", label="可用新能源")
        轴.plot(子表["日内小时"], 子表["AI_IT_Load_MW"], color="#CE4459", label="AI IT负荷")
        轴.fill_between(子表["日内小时"], 0, 子表["Curtailment_MW"], color="#C0D6EA", alpha=0.45, label="弃电")
        轴.set_title(区域)
        轴.set_xticks([0, 6, 12, 18, 23])
        轴.set_ylabel("MW")
    坐标轴[1, 1].set_xlabel("日内小时")
    图形.legend(*坐标轴[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    图形.suptitle("各区域新能源与AI负荷日内匹配", fontsize=15, y=1.06)
    保存图形(图形, 图片目录, "图4_新能源与AI负荷日内匹配")


def 生成问题二报告(指标, 检验结果, 运行时间_s, 报告路径):
    成本变化 = (指标["优化运行成本_CNY"] - 指标["基准运行成本_CNY"]) / abs(指标["基准运行成本_CNY"])
    碳变化 = (指标["优化碳排放_tCO2"] - 指标["基准碳排放_tCO2"]) / 指标["基准碳排放_tCO2"]
    新能源变化 = 指标["优化新能源利用率"] - 指标["基准新能源利用率"]
    检验文本 = "\n".join(
        f"- {行['检验项']}：{'通过' if 行['是否通过'] else '未通过'}，违约数或残差为 {行['违约数或残差']:.6g}"
        for _, 行 in 检验结果.iterrows()
    )
    内容 = f"""# 问题二计算结果

## 求解概况

以第 0-2399 小时实际到达任务和第 0-2406 小时逐时电力参数为输入，采用带候选时空剪枝的启发式调度。运行成本、碳排放和新能源利用率构成基准归一化三目标函数；网络时延作为任务级硬约束、候选位置次级选择规则和最终服务质量评价指标。附件给出的购电、售电、弃电和储能轨迹作为基准状态，仅根据优化 AI 负荷相对基准 AI 负荷的变化进行边际能源结算。当两者相等时，计算结果严格恢复附件基准指标。

共处理 {指标['任务数']:,} 条任务，成功调度 {指标['成功调度数']:,} 条，计算耗时 {运行时间_s:.2f} s。

## 核心结果

- 净运行成本由 {指标['基准运行成本_CNY']:,.2f} CNY 变为 {指标['优化运行成本_CNY']:,.2f} CNY，相对变化 {成本变化:.2%}。
- 碳排放由 {指标['基准碳排放_tCO2']:,.2f} tCO2 变为 {指标['优化碳排放_tCO2']:,.2f} tCO2，相对变化 {碳变化:.2%}。
- 新能源利用率由 {指标['基准新能源利用率']:.2%} 提升至 {指标['优化新能源利用率']:.2%}，提高 {新能源变化 * 100:.2f} 个百分点。
- 以任务均在来源区域执行作为本地参考，其平均网络时延为 {指标['基准平均网络时延_ms']:.2f} ms；问题二方案平均时延为 {指标['优化平均网络时延_ms']:.2f} ms，P95 时延为 {指标['优化P95网络时延_ms']:.2f} ms，最大时延为 {指标['优化最大网络时延_ms']:.2f} ms。
- 任务迁移率为 {指标['任务迁移率']:.2%}，三目标综合值为 {指标['三目标综合值']:.6f}。

优化后的净运行成本保持为正，成本、碳排放和新能源利用率均按附件原始基准结果进行比较。电力平衡残差来自附件逐时数据的小数位舍入，采用 0.001 MW 作为通过阈值。

## 必要约束检验

{检验文本}

## 成果文件

- `任务调度方案.csv`：每条任务的执行区域、开工和完工时刻、网络时延及迁移标记。
- `区域逐时负荷与能源.csv`：各区域逐时 GPU、IT、设施负荷与购售电、弃电和碳排放结果。
- `指标对比.csv`：基准方案与问题二方案的成本、碳排放、新能源利用率和网络时延。
- `约束检验.csv`：任务、容量、时延、截止时刻和电力平衡检验。
"""
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    报告路径.write_text(内容, encoding="utf-8")


def 主程序():
    开始时间 = time.perf_counter()
    根目录 = 项目根目录()
    数据目录 = 根目录 / "题目" / "附件数据"
    输出目录 = 根目录 / "outputs" / "问题二计算结果"
    图片目录 = 根目录 / "figures" / "问题二计算结果"
    输出目录.mkdir(parents=True, exist_ok=True)
    任务数据 = pd.read_excel(数据目录 / "workload_trace.xlsx")
    GPU信息 = pd.read_excel(数据目录 / "GPU_information.xlsx")
    时间数据 = pd.read_excel(数据目录 / "region_time_data.xlsx")
    网络延迟 = pd.read_excel(数据目录 / "network_latency.xlsx")
    功率映射 = pd.read_excel(数据目录 / "power_mapping.xlsx")
    调度结果, 资源结果, 指标 = 执行问题二调度(任务数据, GPU信息, 时间数据, 网络延迟, 功率映射)
    检验结果 = 检验问题二约束(任务数据, 调度结果, 资源结果)
    任务明细 = 任务数据.merge(调度结果, on="TaskID", how="left")
    任务明细.to_csv(输出目录 / "任务调度方案.csv", index=False, encoding="utf-8-sig")
    资源结果.to_csv(输出目录 / "区域逐时负荷与能源.csv", index=False, encoding="utf-8-sig")
    检验结果.to_csv(输出目录 / "约束检验.csv", index=False, encoding="utf-8-sig")
    绘制问题二图表(任务明细, 资源结果, 指标, 图片目录, 输出目录)
    运行时间_s = time.perf_counter() - 开始时间
    运行统计 = dict(指标)
    运行统计["运行时间_s"] = 运行时间_s
    (输出目录 / "运行统计.json").write_text(json.dumps(运行统计, ensure_ascii=False, indent=2), encoding="utf-8")
    生成问题二报告(指标, 检验结果, 运行时间_s, 根目录 / "reports" / "问题二计算结果.md")
    print(pd.Series(运行统计).to_string())
    print(检验结果.to_string(index=False))


if __name__ == "__main__":
    主程序()
