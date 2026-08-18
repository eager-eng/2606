import json
import math
import time
import argparse
from pathlib import Path

import gurobipy as gp
import matplotlib
import numpy as np
import pandas as pd
from gurobipy import GRB
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


统一协调迭代数 = 3


def 项目根目录():
    return Path(__file__).resolve().parents[1]


def 计算小时重叠(开始时刻, 结束时刻, 小时):
    return max(0.0, min(float(结束时刻), 小时 + 1.0) - max(float(开始时刻), float(小时)))


def 读取问题四数据(根目录):
    数据目录 = 根目录 / "题目" / "附件数据"
    问题二目录 = 根目录 / "outputs" / "问题二计算结果"
    任务方案 = pd.read_csv(问题二目录 / "任务调度方案.csv")
    时间数据 = pd.read_excel(数据目录 / "region_time_data.xlsx")
    GPU信息 = pd.read_excel(数据目录 / "GPU_information.xlsx")
    储能信息 = pd.read_excel(数据目录 / "storage_information.xlsx")
    网络延迟 = pd.read_excel(数据目录 / "network_latency.xlsx")
    功率映射 = pd.read_excel(数据目录 / "power_mapping.xlsx")
    任务方案 = 任务方案[任务方案["ArrivalHour"].between(0, 2399)].copy()
    时间数据 = 时间数据[时间数据["Hour"].between(0, 2406)].copy()
    时间数据 = 时间数据.sort_values(["Region", "Hour"]).reset_index(drop=True)
    if len(任务方案) == 0 or not 任务方案["Scheduled"].astype(bool).all():
        raise ValueError("问题二任务方案为空或包含未调度任务")
    小时检查 = 时间数据.groupby("Region")["Hour"].agg(["count", "min", "max"])
    if not ((小时检查["count"] == 2407) & (小时检查["min"] == 0) & (小时检查["max"] == 2406)).all():
        raise ValueError("每个区域必须包含第0至2406小时数据")
    return 任务方案, 时间数据, GPU信息, 储能信息, 网络延迟, 功率映射


def 构造占用时段(开始时刻, 结束时刻):
    return [
        (小时, 计算小时重叠(开始时刻, 结束时刻, 小时))
        for 小时 in range(int(math.floor(开始时刻)), int(math.ceil(结束时刻 - 1e-12)))
    ]


def 重构联合负荷(任务方案, 时间数据, GPU信息, 功率映射, 截止小时=2406):
    区域列表 = GPU信息["Region"].tolist()
    区域序号 = {区域: 序号 for 序号, 区域 in enumerate(区域列表)}
    区域参数 = GPU信息.set_index("Region")
    功率系数 = 功率映射.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()
    GPU占用 = np.zeros((len(区域列表), 截止小时 + 1), dtype=float)
    AI负荷 = np.zeros_like(GPU占用)
    for 行 in 任务方案.itertuples(index=False):
        if not bool(行.Scheduled):
            continue
        区域 = 行.ExecutionRegion
        if 区域 not in 区域序号:
            raise ValueError(f"未知执行区域：{区域}")
        if float(行.FinishHour) > 截止小时 + 1e-9:
            raise ValueError(f"任务{行.TaskID}超过第{截止小时}小时完成")
        r = 区域序号[区域]
        for 小时, 重叠 in 构造占用时段(float(行.StartHour), float(行.FinishHour)):
            if 小时 < 0 or 小时 >= 截止小时:
                raise ValueError(f"任务{行.TaskID}占用非法时段{小时}")
            GPU占用[r, 小时] += float(行.GPU_Demand) * 重叠
            AI负荷[r, 小时] += float(行.GPU_Demand) * float(功率系数[行.TaskType]) * 重叠
    记录 = []
    for 区域 in 区域列表:
        r = 区域序号[区域]
        区域时间 = 时间数据[时间数据["Region"] == 区域].set_index("Hour")
        参数 = 区域参数.loc[区域]
        for 小时 in range(截止小时 + 1):
            时间行 = 区域时间.loc[小时]
            非AI = float(时间行["NonAI_IT_Load_MW"])
            IT负荷 = 非AI + AI负荷[r, 小时]
            设施负荷 = float(参数["PUE"]) * IT负荷
            记录.append(
                {
                    "Region": 区域,
                    "Hour": 小时,
                    "GPU_Used": GPU占用[r, 小时],
                    "AI_IT_Load_MW": AI负荷[r, 小时],
                    "NonAI_IT_Load_MW": 非AI,
                    "IT_Load_MW": IT负荷,
                    "Facility_Load_MW": 设施负荷,
                    "GPU_Utilization": GPU占用[r, 小时] / float(参数["Available_GPU"]),
                    "IT_Utilization": IT负荷 / float(参数["Max_IT_Power_MW"]),
                    "Facility_Utilization": 设施负荷 / float(参数["Max_Facility_Power_MW"]),
                    "AvailableRenewable_MW": float(时间行["AvailableRenewable_MW"]),
                    "ElectricityPrice_CNY_per_MWh": float(时间行["ElectricityPrice_CNY_per_MWh"]),
                    "SellPrice_CNY_per_MWh": float(时间行["SellPrice_CNY_per_MWh"]),
                    "CarbonIntensity_tCO2_per_MWh": float(时间行["CarbonIntensity_tCO2_per_MWh"]),
                }
            )
    return pd.DataFrame(记录)


def 求解能源模型(
    负荷数据,
    储能信息,
    方案名称,
    主目标="成本",
    碳上限=None,
    弃电率上限=None,
    固定模式=None,
    暖启动=None,
    输出日志=False,
    时间限制_s=180.0,
):
    数据 = 负荷数据.sort_values(["Region", "Hour"]).reset_index(drop=True).copy()
    参数表 = 储能信息.set_index("Region")
    数量 = len(数据)
    区域索引 = {区域: 子表.index.to_list() for 区域, 子表 in 数据.groupby("Region", sort=False)}
    购电上限 = {i: float(参数表.loc[数据.at[i, "Region"], "MaxGridImport_MW"]) for i in range(数量)}
    售电上限 = {
        i: min(
            float(参数表.loc[数据.at[i, "Region"], "SellLimit_MW"]),
            float(参数表.loc[数据.at[i, "Region"], "MaxGridExport_MW"]),
        )
        for i in range(数量)
    }
    充电上限 = {i: float(参数表.loc[数据.at[i, "Region"], "MaxChargePower_MW"]) for i in range(数量)}
    放电上限 = {i: float(参数表.loc[数据.at[i, "Region"], "MaxDischargePower_MW"]) for i in range(数量)}
    SOC下限 = {i: float(参数表.loc[数据.at[i, "Region"], "MinSOC_MWh"]) for i in range(数量)}
    SOC上限 = {i: float(参数表.loc[数据.at[i, "Region"], "StorageCapacity_MWh"]) for i in range(数量)}
    负荷 = 数据["Facility_Load_MW"].to_numpy(dtype=float)
    新能源 = 数据["AvailableRenewable_MW"].to_numpy(dtype=float)
    购电价 = 数据["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float)
    售电价 = 数据["SellPrice_CNY_per_MWh"].to_numpy(dtype=float)
    碳强度 = 数据["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float)
    模型 = gp.Model(f"问题四_{方案名称}")
    模型.Params.OutputFlag = 1 if 输出日志 else 0
    模型.Params.TimeLimit = float(时间限制_s)
    模型.Params.MIPGap = 1e-4
    购电 = 模型.addVars(数量, lb=0.0, ub=购电上限, name="购电")
    售电 = 模型.addVars(数量, lb=0.0, ub=售电上限, name="售电")
    充电 = 模型.addVars(数量, lb=0.0, ub=充电上限, name="充电")
    放电 = 模型.addVars(数量, lb=0.0, ub=放电上限, name="放电")
    弃电 = 模型.addVars(数量, lb=0.0, ub={i: float(新能源[i]) for i in range(数量)}, name="弃电")
    SOC = 模型.addVars(数量, lb=SOC下限, ub=SOC上限, name="SOC")
    if 固定模式 is None:
        充电状态 = 模型.addVars(数量, vtype=GRB.BINARY, name="充电状态")
        购电状态 = 模型.addVars(数量, vtype=GRB.BINARY, name="购电状态")
        固定充电状态 = None
        固定购电状态 = None
    else:
        模式表 = 固定模式.set_index(["Region", "Hour"])
        固定充电状态 = np.array(
            [int(round(模式表.loc[(数据.at[i, "Region"], int(数据.at[i, "Hour"])), "ChargeMode"])) for i in range(数量)]
        )
        固定购电状态 = np.array(
            [int(round(模式表.loc[(数据.at[i, "Region"], int(数据.at[i, "Hour"])), "GridBuyMode"])) for i in range(数量)]
        )
        充电状态 = None
        购电状态 = None
    if 暖启动 is not None:
        启动表 = 暖启动.set_index(["Region", "Hour"])
        for i in range(数量):
            键 = (数据.at[i, "Region"], int(数据.at[i, "Hour"]))
            if 键 in 启动表.index:
                行 = 启动表.loc[键]
                购电[i].Start = float(行["GridPurchase_MW"])
                售电[i].Start = float(行["GridSell_MW"])
                充电[i].Start = float(行["ChargePower_MW"])
                放电[i].Start = float(行["DischargePower_MW"])
                弃电[i].Start = float(行["Curtailment_MW"])
                SOC[i].Start = float(行["SOC_MWh"])
                if 充电状态 is not None:
                    充电状态[i].Start = int(round(行["ChargeMode"]))
                    购电状态[i].Start = int(round(行["GridBuyMode"]))
    平衡约束 = {}
    for i in range(数量):
        平衡约束[i] = 模型.addConstr(
            购电[i] + 新能源[i] + 放电[i] == 负荷[i] + 充电[i] + 售电[i] + 弃电[i],
            name=f"设施平衡_{i}",
        )
        模型.addConstr(售电[i] <= 新能源[i] + 放电[i])
        if 固定模式 is None:
            模型.addConstr(充电[i] <= 充电上限[i] * 充电状态[i])
            模型.addConstr(放电[i] <= 放电上限[i] * (1.0 - 充电状态[i]))
            模型.addConstr(购电[i] <= 购电上限[i] * 购电状态[i])
            模型.addConstr(售电[i] <= 售电上限[i] * (1.0 - 购电状态[i]))
        else:
            模型.addConstr(充电[i] <= 充电上限[i] * 固定充电状态[i])
            模型.addConstr(放电[i] <= 放电上限[i] * (1.0 - 固定充电状态[i]))
            模型.addConstr(购电[i] <= 购电上限[i] * 固定购电状态[i])
            模型.addConstr(售电[i] <= 售电上限[i] * (1.0 - 固定购电状态[i]))
    for 区域, 索引列表 in 区域索引.items():
        参数 = 参数表.loc[区域]
        for 位置, i in enumerate(索引列表):
            前一SOC = float(参数["InitialSOC_MWh"]) if 位置 == 0 else SOC[索引列表[位置 - 1]]
            模型.addConstr(
                SOC[i]
                == 前一SOC
                + float(参数["ChargeEfficiency"]) * 充电[i]
                - 放电[i] / float(参数["DischargeEfficiency"])
            )
        模型.addConstr(SOC[索引列表[-1]] >= float(参数["InitialSOC_MWh"]))
    成本表达式 = gp.quicksum(购电[i] * 购电价[i] - 售电[i] * 售电价[i] for i in range(数量))
    碳表达式 = gp.quicksum(购电[i] * 碳强度[i] for i in range(数量))
    弃电表达式 = gp.quicksum(弃电[i] for i in range(数量))
    新能源总量 = max(float(np.sum(新能源)), 1e-12)
    if 碳上限 is not None:
        模型.addConstr(碳表达式 <= float(碳上限), name="碳排放上限")
    if 弃电率上限 is not None:
        模型.addConstr(弃电表达式 <= float(弃电率上限) * 新能源总量, name="弃电率上限")
    if 主目标 == "成本":
        模型.setObjective(成本表达式, GRB.MINIMIZE)
    elif 主目标 == "碳排放":
        模型.ModelSense = GRB.MINIMIZE
        模型.setObjectiveN(碳表达式, 0, priority=2, weight=1.0, name="碳排放")
        模型.setObjectiveN(成本表达式, 1, priority=1, weight=1.0, name="成本")
    elif 主目标 == "弃电":
        模型.ModelSense = GRB.MINIMIZE
        模型.setObjectiveN(弃电表达式, 0, priority=2, weight=1.0, name="弃电")
        模型.setObjectiveN(成本表达式, 1, priority=1, weight=1.0, name="成本")
    else:
        raise ValueError(f"未知主目标：{主目标}")
    模型.optimize()
    状态名称 = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }.get(模型.Status, str(模型.Status))
    try:
        MIP间隙 = float(模型.MIPGap)
    except AttributeError:
        MIP间隙 = 0.0 if 模型.Status == GRB.OPTIMAL else np.nan
    统计 = {
        "方案": 方案名称,
        "求解状态": 状态名称,
        "有可行解": int(模型.SolCount > 0),
        "运行时间_s": float(模型.Runtime),
        "MIPGap": MIP间隙 if 模型.SolCount > 0 and 固定模式 is None else 0.0,
    }
    if 模型.SolCount == 0:
        return None, None, 统计
    结果 = 数据.copy()
    结果["GridPurchase_MW"] = [购电[i].X for i in range(数量)]
    结果["GridSell_MW"] = [售电[i].X for i in range(数量)]
    结果["NetGridImport_MW"] = 结果["GridPurchase_MW"] - 结果["GridSell_MW"]
    结果["ChargePower_MW"] = [充电[i].X for i in range(数量)]
    结果["DischargePower_MW"] = [放电[i].X for i in range(数量)]
    结果["Curtailment_MW"] = [弃电[i].X for i in range(数量)]
    结果["SOC_MWh"] = [SOC[i].X for i in range(数量)]
    if 固定模式 is None:
        结果["ChargeMode"] = [int(round(充电状态[i].X)) for i in range(数量)]
        结果["GridBuyMode"] = [int(round(购电状态[i].X)) for i in range(数量)]
    else:
        结果["ChargeMode"] = 固定充电状态
        结果["GridBuyMode"] = 固定购电状态
    结果["OperatingCost_CNY"] = 结果["GridPurchase_MW"] * 购电价 - 结果["GridSell_MW"] * 售电价
    结果["CarbonEmission_tCO2"] = 结果["GridPurchase_MW"] * 碳强度
    结果["PowerBalanceResidual_MW"] = (
        结果["GridPurchase_MW"]
        + 新能源
        + 结果["DischargePower_MW"]
        - 负荷
        - 结果["ChargePower_MW"]
        - 结果["GridSell_MW"]
        - 结果["Curtailment_MW"]
    )
    结果["Scenario"] = 方案名称
    指标 = 计算能源指标(结果)
    统计.update(指标)
    if 固定模式 is not None:
        结果["ConditionalMarginalEnergy_CNY_per_MWh"] = [平衡约束[i].Pi for i in range(数量)]
    return 结果, 指标, 统计


def 计算能源指标(逐时结果):
    新能源总量 = float(逐时结果["AvailableRenewable_MW"].sum())
    弃电量 = float(逐时结果["Curtailment_MW"].sum())
    区域峰值 = (
        逐时结果.assign(正向净购电=np.maximum(逐时结果["NetGridImport_MW"], 0.0))
        .groupby("Region")["正向净购电"]
        .max()
        .to_dict()
    )
    return {
        "OperatingCost_CNY": float(逐时结果["OperatingCost_CNY"].sum()),
        "CarbonEmission_tCO2": float(逐时结果["CarbonEmission_tCO2"].sum()),
        "Curtailment_MWh": 弃电量,
        "CurtailmentRate": 弃电量 / 新能源总量 if 新能源总量 > 0 else 0.0,
        "RenewableUtilization": 1.0 - 弃电量 / 新能源总量 if 新能源总量 > 0 else 0.0,
        "PeakByRegion": 区域峰值,
    }


def 删除支配解(方案表, 容差=1e-8):
    数据 = 方案表.reset_index(drop=True).copy()
    指标列 = ["OperatingCost_CNY", "CarbonEmission_tCO2", "CurtailmentRate"]
    保留 = []
    for i, 行 in 数据.iterrows():
        当前 = 行[指标列].to_numpy(dtype=float)
        被支配 = False
        for j, 其他 in 数据.iterrows():
            if i == j:
                continue
            对方 = 其他[指标列].to_numpy(dtype=float)
            if np.all(对方 <= 当前 + 容差) and np.any(对方 < 当前 - 容差):
                被支配 = True
                break
        保留.append(not 被支配)
    return 数据.loc[保留].reset_index(drop=True)


def 选择理想点方案(方案表):
    数据 = 方案表.copy()
    映射 = {
        "OperatingCost_CNY": "NormalizedCost",
        "CarbonEmission_tCO2": "NormalizedCarbon",
        "CurtailmentRate": "NormalizedCurtailment",
    }
    for 原列, 新列 in 映射.items():
        最小值 = float(数据[原列].min())
        最大值 = float(数据[原列].max())
        数据[新列] = 0.0 if 最大值 - 最小值 <= 1e-12 else (数据[原列] - 最小值) / (最大值 - 最小值)
    数据["IdealPointDistance"] = np.sqrt(
        数据["NormalizedCost"] ** 2 + 数据["NormalizedCarbon"] ** 2 + 数据["NormalizedCurtailment"] ** 2
    )
    数据["Selected"] = False
    选中索引 = 数据["IdealPointDistance"].idxmin()
    数据.loc[选中索引, "Selected"] = True
    return 数据, 数据.loc[选中索引]


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


def 计算任务能源得分(行, 区域, 开工, 信号矩阵, 区域序号, PUE映射, 功率系数):
    完工 = float(开工) + float(行["EstimatedDuration_min"]) / 60.0
    r = 区域序号[区域]
    单位功率 = float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * float(PUE映射[区域])
    return float(
        sum(信号矩阵[r, 小时] * 单位功率 * 重叠 for 小时, 重叠 in 构造占用时段(开工, 完工))
    )


def 调整弹性任务(
    任务方案,
    负荷数据,
    影子价格结果,
    GPU信息,
    储能信息,
    网络延迟,
    功率映射,
    调整比例=0.05,
    候选数=3,
    候选池上限=None,
    截止小时=2406,
):
    任务 = 任务方案.copy().reset_index(drop=True)
    区域列表 = GPU信息["Region"].tolist()
    区域序号 = {区域: 序号 for 序号, 区域 in enumerate(区域列表)}
    区域参数 = GPU信息.set_index("Region")
    储能参数 = 储能信息.set_index("Region")
    PUE映射 = 区域参数["PUE"].to_dict()
    功率系数 = 功率映射.set_index("TaskType")["GPU_Power_MW_per_EquivalentGPU"].to_dict()
    延迟映射 = 网络延迟.set_index(["FromRegion", "ToRegion"])["NetworkLatency_ms"].to_dict()

    def 透视(数据, 字段):
        return (
            数据.pivot(index="Hour", columns="Region", values=字段)
            .reindex(index=range(截止小时 + 1), columns=区域列表)
            .to_numpy(dtype=float)
            .T
            .copy()
        )

    GPU占用 = 透视(负荷数据, "GPU_Used")
    AI负荷 = 透视(负荷数据, "AI_IT_Load_MW")
    非AI负荷 = 透视(负荷数据, "NonAI_IT_Load_MW")
    新能源 = 透视(负荷数据, "AvailableRenewable_MW")
    信号矩阵 = 透视(影子价格结果, "ConditionalMarginalEnergy_CNY_per_MWh")
    候选器 = [区间候选器(信号矩阵[r, :截止小时], 候选数) for r in range(len(区域列表))]
    弹性索引 = 任务.index[任务["TaskType"].isin(["BatchInference", "AITraining"])].tolist()
    if 候选池上限 is not None and len(弹性索引) > int(候选池上限):
        候选能耗 = (
            任务.loc[弹性索引, "GPU_Demand"]
            * 任务.loc[弹性索引, "EstimatedDuration_min"]
            * 任务.loc[弹性索引, "TaskType"].map(功率系数)
        )
        弹性索引 = 候选能耗.nlargest(int(候选池上限)).index.tolist()
    最大调整数 = max(1, int(math.ceil(len(弹性索引) * 调整比例)))
    改进候选 = []

    def 评估候选(索引, 区域, 开工):
        行 = 任务.loc[索引]
        时长 = float(行["EstimatedDuration_min"]) / 60.0
        完工 = float(开工) + 时长
        if float(开工) < max(float(行["ArrivalHour"]), float(行["EarliestStartHour"])) - 1e-9:
            return None
        if 完工 > min(float(行["LatestFinishHour"]), 截止小时) + 1e-9:
            return None
        延迟 = float(延迟映射.get((行["SourceRegion"], 区域), np.inf))
        if 延迟 > float(行["MaxLatency_ms"]) + 1e-9:
            return None
        r = 区域序号[区域]
        当前区域 = 行["ExecutionRegion"]
        当前占用 = dict(构造占用时段(float(行["StartHour"]), float(行["FinishHour"])))
        候选占用 = 构造占用时段(float(开工), 完工)
        小时 = np.array([值[0] for 值 in 候选占用], dtype=int)
        重叠 = np.array([值[1] for 值 in 候选占用], dtype=float)
        if len(小时) == 0 or np.any(小时 < 0) or np.any(小时 >= 截止小时):
            return None
        当前扣减 = np.array([当前占用.get(int(h), 0.0) if 区域 == 当前区域 else 0.0 for h in 小时])
        GPU增量 = float(行["GPU_Demand"]) * (重叠 - 当前扣减)
        AI增量 = float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * (重叠 - 当前扣减)
        新GPU = GPU占用[r, 小时] + GPU增量
        新AI = AI负荷[r, 小时] + AI增量
        新IT = 非AI负荷[r, 小时] + 新AI
        新设施 = float(PUE映射[区域]) * 新IT
        if np.any(新GPU > float(区域参数.loc[区域, "Available_GPU"]) + 1e-9):
            return None
        if np.any(新IT > float(区域参数.loc[区域, "Max_IT_Power_MW"]) + 1e-9):
            return None
        if np.any(新设施 > float(区域参数.loc[区域, "Max_Facility_Power_MW"]) + 1e-9):
            return None
        最大供能 = (
            float(储能参数.loc[区域, "MaxGridImport_MW"])
            + 新能源[r, 小时]
            + float(储能参数.loc[区域, "MaxDischargePower_MW"])
        )
        if np.any(新设施 > 最大供能 + 1e-9):
            return None
        候选得分 = 计算任务能源得分(行, 区域, 开工, 信号矩阵, 区域序号, PUE映射, 功率系数)
        当前得分 = 计算任务能源得分(
            行,
            当前区域,
            float(行["StartHour"]),
            信号矩阵,
            区域序号,
            PUE映射,
            功率系数,
        )
        return {
            "索引": 索引,
            "Region": 区域,
            "StartHour": float(开工),
            "FinishHour": 完工,
            "NetworkLatency_ms": 延迟,
            "EnergySaving_CNY": 当前得分 - 候选得分,
        }

    for 索引 in 弹性索引:
        行 = 任务.loc[索引]
        时长 = float(行["EstimatedDuration_min"]) / 60.0
        最早 = int(math.ceil(max(float(行["ArrivalHour"]), float(行["EarliestStartHour"]))))
        最晚 = int(math.floor(min(float(行["LatestFinishHour"]), 截止小时) - 时长 + 1e-10))
        if 最晚 < 最早:
            continue
        任务候选 = []
        for 区域 in 区域列表:
            延迟 = float(延迟映射.get((行["SourceRegion"], 区域), np.inf))
            if 延迟 > float(行["MaxLatency_ms"]) + 1e-9:
                continue
            r = 区域序号[区域]
            开工集合 = set(候选器[r].查询(最早, 最晚) + [最早, 最晚])
            if 区域 == 行["ExecutionRegion"]:
                开工集合.add(int(round(float(行["StartHour"]))))
            for 开工 in sorted(开工集合):
                if 区域 == 行["ExecutionRegion"] and abs(float(开工) - float(行["StartHour"])) <= 1e-9:
                    continue
                候选 = 评估候选(索引, 区域, 开工)
                if 候选 is not None:
                    任务候选.append(候选)
        if 任务候选:
            最优 = min(
                任务候选,
                key=lambda x: (-x["EnergySaving_CNY"], x["NetworkLatency_ms"], x["StartHour"], x["Region"]),
            )
            if 最优["EnergySaving_CNY"] > 1e-6:
                最优["LatestFinishHour"] = float(行["LatestFinishHour"])
                改进候选.append(最优)
    改进候选 = sorted(改进候选, key=lambda x: -x["EnergySaving_CNY"])[:最大调整数]
    改进候选 = sorted(改进候选, key=lambda x: (x["LatestFinishHour"], -x["EnergySaving_CNY"]))
    已调整 = []
    for 候选 in 改进候选:
        索引 = 候选["索引"]
        最新评估 = 评估候选(索引, 候选["Region"], 候选["StartHour"])
        if 最新评估 is None or 最新评估["EnergySaving_CNY"] <= 1e-6:
            continue
        行 = 任务.loc[索引]
        当前区域 = 行["ExecutionRegion"]
        当前r = 区域序号[当前区域]
        for 小时, 重叠 in 构造占用时段(float(行["StartHour"]), float(行["FinishHour"])):
            GPU占用[当前r, 小时] -= float(行["GPU_Demand"]) * 重叠
            AI负荷[当前r, 小时] -= float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 重叠
        新r = 区域序号[最新评估["Region"]]
        for 小时, 重叠 in 构造占用时段(最新评估["StartHour"], 最新评估["FinishHour"]):
            GPU占用[新r, 小时] += float(行["GPU_Demand"]) * 重叠
            AI负荷[新r, 小时] += float(行["GPU_Demand"]) * float(功率系数[行["TaskType"]]) * 重叠
        任务.at[索引, "ExecutionRegion"] = 最新评估["Region"]
        任务.at[索引, "StartHour"] = 最新评估["StartHour"]
        任务.at[索引, "FinishHour"] = 最新评估["FinishHour"]
        任务.at[索引, "NetworkLatency_ms"] = 最新评估["NetworkLatency_ms"]
        任务.at[索引, "MigrationFlag"] = int(最新评估["Region"] != 行["SourceRegion"])
        已调整.append(
            {
                "TaskID": 行["TaskID"],
                "OriginalRegion": 当前区域,
                "NewRegion": 最新评估["Region"],
                "OriginalStartHour": float(行["StartHour"]),
                "NewStartHour": 最新评估["StartHour"],
                "EstimatedEnergySaving_CNY": 最新评估["EnergySaving_CNY"],
            }
        )
    return 任务, pd.DataFrame(已调整)


def 协调任务与能源(
    初始任务,
    时间数据,
    GPU信息,
    储能信息,
    网络延迟,
    功率映射,
    方案名称,
    碳上限,
    弃电率上限,
    初始能源=None,
    最大迭代=3,
    候选池上限=None,
    输出日志=False,
    时间限制_s=180.0,
):
    当前任务 = 初始任务.copy()
    当前负荷 = 重构联合负荷(当前任务, 时间数据, GPU信息, 功率映射)
    if 初始能源 is None:
        当前能源, 当前指标, 当前统计 = 求解能源模型(
            当前负荷,
            储能信息,
            方案名称,
            主目标="成本",
            碳上限=碳上限,
            弃电率上限=弃电率上限,
            输出日志=输出日志,
            时间限制_s=时间限制_s,
        )
    else:
        当前能源 = 初始能源.copy()
        当前指标 = 计算能源指标(当前能源)
        当前统计 = {"方案": 方案名称, "求解状态": "REUSED", "有可行解": 1, "运行时间_s": 0.0, "MIPGap": 0.0}
    求解统计 = [当前统计]
    迭代记录 = []
    迁移明细 = []
    if 当前能源 is None:
        return 当前任务, 当前负荷, None, None, pd.DataFrame(迭代记录), pd.DataFrame(迁移明细), pd.DataFrame(求解统计)
    for 迭代 in range(1, 最大迭代 + 1):
        影子结果, _, 影子统计 = 求解能源模型(
            当前负荷,
            储能信息,
            f"{方案名称}_影子价格_{迭代}",
            主目标="成本",
            碳上限=碳上限,
            弃电率上限=弃电率上限,
            固定模式=当前能源,
            暖启动=当前能源,
            输出日志=False,
            时间限制_s=时间限制_s,
        )
        求解统计.append(影子统计)
        if 影子结果 is None:
            break
        候选任务, 本轮迁移 = 调整弹性任务(
            当前任务,
            当前负荷,
            影子结果,
            GPU信息,
            储能信息,
            网络延迟,
            功率映射,
            候选池上限=候选池上限,
        )
        if len(本轮迁移) == 0:
            迭代记录.append(
                {"Scenario": 方案名称, "Iteration": 迭代, "MovedTasks": 0, "Accepted": False, "CostImprovementRate": 0.0}
            )
            break
        候选负荷 = 重构联合负荷(候选任务, 时间数据, GPU信息, 功率映射)
        候选能源, 候选指标, 候选统计 = 求解能源模型(
            候选负荷,
            储能信息,
            f"{方案名称}_协调_{迭代}",
            主目标="成本",
            碳上限=碳上限,
            弃电率上限=弃电率上限,
            暖启动=当前能源,
            输出日志=输出日志,
            时间限制_s=时间限制_s,
        )
        求解统计.append(候选统计)
        原成本 = float(当前指标["OperatingCost_CNY"])
        新成本 = float(候选指标["OperatingCost_CNY"]) if 候选指标 is not None else np.inf
        改善率 = (原成本 - 新成本) / max(abs(原成本), 1e-12)
        接受 = 候选能源 is not None and 新成本 < 原成本 - 1e-6
        迭代记录.append(
            {
                "Scenario": 方案名称,
                "Iteration": 迭代,
                "MovedTasks": int(len(本轮迁移)),
                "Accepted": bool(接受),
                "CostBefore_CNY": 原成本,
                "CostAfter_CNY": 新成本 if np.isfinite(新成本) else np.nan,
                "CostImprovementRate": 改善率 if np.isfinite(改善率) else np.nan,
            }
        )
        if not 接受:
            break
        本轮迁移["Scenario"] = 方案名称
        本轮迁移["Iteration"] = 迭代
        迁移明细.extend(本轮迁移.to_dict("records"))
        当前任务 = 候选任务
        当前负荷 = 候选负荷
        当前能源 = 候选能源
        当前指标 = 候选指标
        if 改善率 < 0.001:
            break
    return (
        当前任务,
        当前负荷,
        当前能源,
        当前指标,
        pd.DataFrame(迭代记录),
        pd.DataFrame(迁移明细),
        pd.DataFrame(求解统计),
    )


def 构造Pareto前沿(负荷数据, 储能信息, 输出日志=False, 时间限制_s=180.0):
    端点设置 = [
        ("最低成本端点", "成本", "最低成本端点"),
        ("最低碳排放端点", "碳排放", "最低碳排放端点"),
        ("最高新能源利用率端点", "弃电", "最高新能源利用率端点"),
    ]
    结果映射 = {}
    指标记录 = []
    求解统计 = []
    上一结果 = None
    for 方案名称, 主目标, 方案类型 in 端点设置:
        结果, 指标, 统计 = 求解能源模型(
            负荷数据,
            储能信息,
            方案名称,
            主目标=主目标,
            暖启动=上一结果,
            输出日志=输出日志,
            时间限制_s=时间限制_s,
        )
        求解统计.append(统计)
        if 结果 is None:
            raise RuntimeError(f"{方案名称}未得到可行解")
        print(f"{方案名称}完成", flush=True)
        上一结果 = 结果
        结果映射[方案名称] = 结果
        指标记录.append(
            {
                "方案": 方案名称,
                "方案类型": 方案类型,
                "EpsilonCarbon_tCO2": np.nan,
                "EpsilonCurtailmentRate": np.nan,
                **{键: 值 for 键, 值 in 指标.items() if 键 != "PeakByRegion"},
            }
        )
    端点表 = pd.DataFrame(指标记录)
    碳最小 = float(端点表["CarbonEmission_tCO2"].min())
    碳最大 = float(端点表["CarbonEmission_tCO2"].max())
    弃电最小 = float(端点表["CurtailmentRate"].min())
    弃电最大 = float(端点表["CurtailmentRate"].max())
    碳层级 = np.linspace(碳最小, 碳最大, 3)
    弃电层级 = np.linspace(弃电最小, 弃电最大, 3)
    for 碳序号, 碳上限 in enumerate(碳层级, 1):
        for 弃电序号, 弃电上限 in enumerate(弃电层级, 1):
            方案名称 = f"Pareto_E{碳序号}_Q{弃电序号}"
            结果, 指标, 统计 = 求解能源模型(
                负荷数据,
                储能信息,
                方案名称,
                主目标="成本",
                碳上限=float(碳上限) + 1e-7,
                弃电率上限=float(弃电上限) + 1e-10,
                暖启动=结果映射.get("最低成本端点"),
                输出日志=输出日志,
                时间限制_s=时间限制_s,
            )
            求解统计.append(统计)
            if 结果 is None:
                print(f"{方案名称}不可行，已跳过", flush=True)
                continue
            print(f"{方案名称}完成", flush=True)
            结果映射[方案名称] = 结果
            指标记录.append(
                {
                    "方案": 方案名称,
                    "方案类型": "Pareto方案",
                    "EpsilonCarbon_tCO2": float(碳上限),
                    "EpsilonCurtailmentRate": float(弃电上限),
                    **{键: 值 for 键, 值 in 指标.items() if 键 != "PeakByRegion"},
                }
            )
    全部方案 = pd.DataFrame(指标记录)
    去重键 = 全部方案[["OperatingCost_CNY", "CarbonEmission_tCO2", "CurtailmentRate"]].round(7)
    全部方案 = 全部方案.loc[~去重键.duplicated()].reset_index(drop=True)
    非支配方案 = 删除支配解(全部方案)
    非支配方案, 选中方案 = 选择理想点方案(非支配方案)
    范围 = {
        "CarbonMin_tCO2": 碳最小,
        "CarbonMax_tCO2": 碳最大,
        "CurtailmentMinRate": 弃电最小,
        "CurtailmentMaxRate": 弃电最大,
    }
    return 非支配方案, 选中方案, 结果映射, pd.DataFrame(求解统计), 范围


def 提取方案约束(方案行):
    碳上限 = (
        float(方案行["EpsilonCarbon_tCO2"])
        if pd.notna(方案行["EpsilonCarbon_tCO2"])
        else float(方案行["CarbonEmission_tCO2"]) + 1e-6
    )
    弃电上限 = (
        float(方案行["EpsilonCurtailmentRate"])
        if pd.notna(方案行["EpsilonCurtailmentRate"])
        else float(方案行["CurtailmentRate"]) + 1e-10
    )
    return 碳上限, 弃电上限


def 联合协调Pareto方案(
    Pareto表,
    初始能源映射,
    初始任务,
    时间数据,
    GPU信息,
    储能信息,
    网络延迟,
    功率映射,
    输出日志=False,
    时间限制_s=180.0,
):
    记录 = []
    结果映射 = {}
    迭代列表 = []
    迁移列表 = []
    统计列表 = []
    for _, 方案行 in Pareto表.iterrows():
        名称 = str(方案行["方案"])
        碳上限, 弃电上限 = 提取方案约束(方案行)
        任务, 负荷, 能源, 指标, 迭代, 迁移, 统计 = 协调任务与能源(
            初始任务,
            时间数据,
            GPU信息,
            储能信息,
            网络延迟,
            功率映射,
            f"{名称}_联合协调",
            碳上限,
            弃电上限,
            初始能源=初始能源映射[名称],
            最大迭代=统一协调迭代数,
            输出日志=输出日志,
            时间限制_s=时间限制_s,
        )
        if 能源 is None:
            continue
        任务指标 = 计算任务指标(任务)
        记录.append(
            {
                "方案": 名称,
                "方案类型": "联合协调方案",
                "EpsilonCarbon_tCO2": 碳上限,
                "EpsilonCurtailmentRate": 弃电上限,
                "InitialOperatingCost_CNY": float(方案行["OperatingCost_CNY"]),
                "InitialCarbonEmission_tCO2": float(方案行["CarbonEmission_tCO2"]),
                "InitialRenewableUtilization": float(方案行["RenewableUtilization"]),
                **{键: 值 for 键, 值 in 指标.items() if 键 != "PeakByRegion"},
                **任务指标,
                "MaxRegionalPeak_MW": max(指标["PeakByRegion"].values()),
            }
        )
        结果映射[名称] = {
            "名称": 名称,
            "任务": 任务,
            "负荷": 负荷,
            "能源": 能源,
            "指标": 指标,
            "迭代": 迭代,
            "迁移": 迁移,
            "统计": 统计,
            "碳上限": 碳上限,
            "弃电上限": 弃电上限,
        }
        迭代列表.append(迭代)
        迁移列表.append(迁移)
        统计列表.append(统计)
    协调表 = pd.DataFrame(记录)
    if len(协调表) == 0:
        raise RuntimeError("五个Pareto方案联合协调后均无可行解")
    非支配表 = 删除支配解(协调表)
    非支配名称 = set(非支配表["方案"])
    协调表["CoordinatedNonDominated"] = 协调表["方案"].isin(非支配名称)
    for 原列, 新列 in {
        "OperatingCost_CNY": "NormalizedCost",
        "CarbonEmission_tCO2": "NormalizedCarbon",
        "CurtailmentRate": "NormalizedCurtailment",
    }.items():
        最小值 = float(非支配表[原列].min())
        最大值 = float(非支配表[原列].max())
        协调表[新列] = 0.0 if 最大值 - 最小值 <= 1e-12 else (协调表[原列] - 最小值) / (最大值 - 最小值)
    协调表["IdealPointDistance"] = np.sqrt(
        协调表["NormalizedCost"] ** 2
        + 协调表["NormalizedCarbon"] ** 2
        + 协调表["NormalizedCurtailment"] ** 2
    )
    协调表["Selected"] = False
    候选索引 = 协调表.index[协调表["CoordinatedNonDominated"]]
    选中索引 = 协调表.loc[候选索引, "IdealPointDistance"].idxmin()
    协调表.loc[选中索引, "Selected"] = True
    选中行 = 协调表.loc[选中索引]
    return (
        协调表,
        选中行,
        结果映射,
        pd.concat(迭代列表, ignore_index=True) if 迭代列表 else pd.DataFrame(),
        pd.concat(迁移列表, ignore_index=True) if 迁移列表 else pd.DataFrame(),
        pd.concat(统计列表, ignore_index=True) if 统计列表 else pd.DataFrame(),
    )


def 构造电价情景(时间数据, 系数):
    数据 = 时间数据.copy()
    均价 = 数据.groupby("Region")["ElectricityPrice_CNY_per_MWh"].transform("mean")
    数据["ElectricityPrice_CNY_per_MWh"] = np.maximum(
        0.0,
        均价 + float(系数) * (数据["ElectricityPrice_CNY_per_MWh"] - 均价),
    )
    return 数据


def 构造新能源波动情景(时间数据, 系数):
    数据 = 时间数据.sort_values(["Region", "Hour"]).copy()
    平滑值 = (
        数据.groupby("Region", group_keys=False)["AvailableRenewable_MW"]
        .apply(lambda x: x.rolling(24, center=True, min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .reindex(数据.index)
    )
    区域上限 = 数据.groupby("Region")["AvailableRenewable_MW"].transform("max")
    数据["AvailableRenewable_MW"] = np.maximum(
        0.0,
        np.minimum(
            区域上限,
            平滑值 + float(系数) * (数据["AvailableRenewable_MW"] - 平滑值),
        ),
    )
    return 数据.sort_values(["Region", "Hour"]).reset_index(drop=True)


def 计算任务指标(任务方案):
    等待时间 = 任务方案["StartHour"] - 任务方案["ArrivalHour"]
    实时 = 任务方案["TaskType"] == "RealTimeInference"
    立即开工率 = float((np.abs(等待时间[实时]) <= 1e-9).mean()) if 实时.any() else 1.0
    按时 = 任务方案["FinishHour"] <= np.minimum(任务方案["LatestFinishHour"], 2406) + 1e-9
    return {
        "AverageLatency_ms": float(任务方案["NetworkLatency_ms"].mean()),
        "P95Latency_ms": float(任务方案["NetworkLatency_ms"].quantile(0.95)),
        "MaxLatency_ms": float(任务方案["NetworkLatency_ms"].max()),
        "ImmediateStartRate": 立即开工率,
        "OnTimeCompletionRate": float(按时.mean()),
        "AverageWaitingHour": float(等待时间.mean()),
        "P95WaitingHour": float(等待时间.quantile(0.95)),
        "MigrationRate": float(任务方案["MigrationFlag"].mean()),
    }


def 概括任务变化(基准任务, 情景任务):
    对照 = 基准任务[["TaskID", "ExecutionRegion", "StartHour"]].rename(
        columns={"ExecutionRegion": "BaseRegion", "StartHour": "BaseStartHour"}
    )
    合并 = 情景任务.merge(对照, on="TaskID", how="left")
    改变 = 合并[
        (合并["ExecutionRegion"] != 合并["BaseRegion"])
        | (np.abs(合并["StartHour"] - 合并["BaseStartHour"]) > 1e-9)
    ]
    区域改变 = 改变[改变["ExecutionRegion"] != 改变["BaseRegion"]]
    if len(区域改变):
        方向 = 区域改变.groupby(["BaseRegion", "ExecutionRegion"]).size().idxmax()
        主要方向 = f"{方向[0]}->{方向[1]}"
    else:
        主要方向 = "无跨区变化"
    return int(len(改变)), 主要方向


def 计算储能策略(能源结果):
    充电量 = float(能源结果["ChargePower_MW"].sum())
    放电量 = float(能源结果["DischargePower_MW"].sum())
    小时 = (能源结果["Hour"].to_numpy(dtype=float) % 24.0)
    充电权重 = 能源结果["ChargePower_MW"].to_numpy(dtype=float)
    放电权重 = 能源结果["DischargePower_MW"].to_numpy(dtype=float)
    充电时刻 = float(np.average(小时, weights=充电权重)) if 充电量 > 1e-12 else np.nan
    放电时刻 = float(np.average(小时, weights=放电权重)) if 放电量 > 1e-12 else np.nan
    return {
        "ChargeEnergy_MWh": 充电量,
        "DischargeEnergy_MWh": 放电量,
        "WeightedChargeHour": 充电时刻,
        "WeightedDischargeHour": 放电时刻,
    }


def 运行单因素情景(
    统一初始任务,
    基准时间数据,
    GPU信息,
    储能信息,
    网络延迟,
    功率映射,
    基准碳上限,
    基准弃电上限,
    Pareto范围,
    输出日志=False,
    时间限制_s=180.0,
):
    碳最小 = Pareto范围["CarbonMin_tCO2"]
    碳最大 = Pareto范围["CarbonMax_tCO2"]
    情景设置 = [
        ("宽松碳约束", 基准时间数据.copy(), 碳最小 + 0.75 * (碳最大 - 碳最小), 基准弃电上限),
        ("严格碳约束", 基准时间数据.copy(), 碳最小 + 0.25 * (碳最大 - 碳最小), 基准弃电上限),
        ("平坦电价", 构造电价情景(基准时间数据, 0.0), 基准碳上限, 基准弃电上限),
        ("扩大峰谷价差", 构造电价情景(基准时间数据, 1.5), 基准碳上限, 基准弃电上限),
        ("低新能源波动", 构造新能源波动情景(基准时间数据, 0.5), 基准碳上限, 基准弃电上限),
        ("高新能源波动", 构造新能源波动情景(基准时间数据, 1.5), 基准碳上限, 基准弃电上限),
    ]
    所有结果 = []
    for 名称, 时间数据, 碳上限, 弃电上限 in 情景设置:
        任务, 负荷, 能源, 指标, 迭代, 迁移, 统计 = 协调任务与能源(
            统一初始任务,
            时间数据,
            GPU信息,
            储能信息,
            网络延迟,
            功率映射,
            名称,
            碳上限,
            弃电上限,
            最大迭代=统一协调迭代数,
            候选池上限=10000,
            输出日志=输出日志,
            时间限制_s=时间限制_s,
        )
        if 能源 is None and "新能源波动" in 名称:
            初始负荷 = 重构联合负荷(统一初始任务, 时间数据, GPU信息, 功率映射)
            最低碳结果, 最低碳指标, 最低碳统计 = 求解能源模型(
                初始负荷,
                储能信息,
                f"{名称}_最低碳端点",
                主目标="碳排放",
                输出日志=输出日志,
                时间限制_s=时间限制_s,
            )
            if 最低碳指标 is not None:
                碳上限 = float(最低碳指标["CarbonEmission_tCO2"]) + 1e-6
            最低弃电结果, 最低弃电指标, 最低弃电统计 = 求解能源模型(
                初始负荷,
                储能信息,
                f"{名称}_最低弃电端点",
                主目标="弃电",
                碳上限=碳上限,
                输出日志=输出日志,
                时间限制_s=时间限制_s,
            )
            成本端点结果, 成本端点指标, 成本端点统计 = 求解能源模型(
                初始负荷,
                储能信息,
                f"{名称}_成本端点",
                主目标="成本",
                碳上限=碳上限,
                输出日志=输出日志,
                时间限制_s=时间限制_s,
            )
            统计 = pd.concat([统计, pd.DataFrame([最低碳统计, 最低弃电统计, 成本端点统计])], ignore_index=True)
            if 最低弃电指标 is not None and 成本端点指标 is not None:
                弃电上限 = 0.5 * (最低弃电指标["CurtailmentRate"] + 成本端点指标["CurtailmentRate"])
                任务, 负荷, 能源, 指标, 迭代, 迁移, 重算统计 = 协调任务与能源(
                    统一初始任务,
                    时间数据,
                    GPU信息,
                    储能信息,
                    网络延迟,
                    功率映射,
                    名称,
                    碳上限,
                    弃电上限,
                    最大迭代=统一协调迭代数,
                    候选池上限=10000,
                    输出日志=输出日志,
                    时间限制_s=时间限制_s,
                )
                统计 = pd.concat([统计, 重算统计], ignore_index=True)
        所有结果.append(
            {
                "名称": 名称,
                "任务": 任务,
                "负荷": 负荷,
                "能源": 能源,
                "指标": 指标,
                "迭代": 迭代,
                "迁移": 迁移,
                "统计": 统计,
                "碳上限": 碳上限,
                "弃电上限": 弃电上限,
            }
        )
    return 所有结果


def 构造核心结果表(Pareto表, 基准任务, 基准能源, 情景结果):
    基准指标 = 计算能源指标(基准能源)
    Pareto输出 = Pareto表.copy()
    Pareto输出 = Pareto输出[
        [
            "方案",
            "方案类型",
            "InitialOperatingCost_CNY",
            "OperatingCost_CNY",
            "CarbonEmission_tCO2",
            "RenewableUtilization",
            "AverageLatency_ms",
            "P95Latency_ms",
            "OnTimeCompletionRate",
            "MaxRegionalPeak_MW",
            "CoordinatedNonDominated",
            "IdealPointDistance",
            "Selected",
        ]
    ].rename(
        columns={
            "InitialOperatingCost_CNY": "协调前运行成本_CNY",
            "OperatingCost_CNY": "运行成本_CNY",
            "CarbonEmission_tCO2": "碳排放_tCO2",
            "RenewableUtilization": "新能源利用率",
            "AverageLatency_ms": "平均网络时延_ms",
            "P95Latency_ms": "P95网络时延_ms",
            "OnTimeCompletionRate": "按时完成率",
            "MaxRegionalPeak_MW": "最大区域峰值_MW",
            "CoordinatedNonDominated": "协调后非支配",
            "IdealPointDistance": "理想点距离",
            "Selected": "是否选中",
        }
    )
    任务指标 = 计算任务指标(基准任务)
    综合记录 = [
        ("系统净运行成本", 基准指标["OperatingCost_CNY"], "CNY"),
        ("系统碳排放量", 基准指标["CarbonEmission_tCO2"], "tCO2"),
        ("新能源利用率", 基准指标["RenewableUtilization"] * 100.0, "%"),
        ("平均网络时延", 任务指标["AverageLatency_ms"], "ms"),
        ("P95网络时延", 任务指标["P95Latency_ms"], "ms"),
        ("最大网络时延", 任务指标["MaxLatency_ms"], "ms"),
        ("任务迁移率", 任务指标["MigrationRate"] * 100.0, "%"),
        ("实时任务立即开工率", 任务指标["ImmediateStartRate"] * 100.0, "%"),
        ("全部任务按时完成率", 任务指标["OnTimeCompletionRate"] * 100.0, "%"),
        ("平均等待时间", 任务指标["AverageWaitingHour"], "h"),
        ("P95等待时间", 任务指标["P95WaitingHour"], "h"),
    ]
    for 区域, 峰值 in 基准指标["PeakByRegion"].items():
        综合记录.append((f"{区域}峰值净购电功率", 峰值, "MW"))
    基准综合表 = pd.DataFrame(综合记录, columns=["指标", "结果", "单位"])
    基准储能 = 计算储能策略(基准能源)
    情景记录 = []
    全部情景 = [
        {
            "名称": "基准折中",
            "任务": 基准任务,
            "能源": 基准能源,
            "指标": 基准指标,
        }
    ] + 情景结果
    for 情景 in 全部情景:
        名称 = 情景["名称"]
        能源 = 情景["能源"]
        指标 = 情景["指标"]
        if 能源 is None or 指标 is None:
            情景记录.append(
                {
                    "情景": 名称,
                    "运行成本_CNY": np.nan,
                    "碳排放_tCO2": np.nan,
                    "新能源利用率": np.nan,
                    "六区域峰值净购电功率_MW": "不可行",
                    "任务调度变化": "不可行",
                    "储能策略变化": "不可行",
                    "平均网络时延_ms": np.nan,
                    "P95网络时延_ms": np.nan,
                    "最大网络时延_ms": np.nan,
                    "迁移率_%": np.nan,
                    "实时任务立即开工率_%": np.nan,
                    "按时完成率_%": np.nan,
                    "平均等待时间_h": np.nan,
                    "P95等待时间_h": np.nan,
                }
            )
            continue
        任务变化数, 主要方向 = 概括任务变化(基准任务, 情景["任务"])
        任务指标 = 计算任务指标(情景["任务"])
        储能策略 = 计算储能策略(能源)
        峰值向量 = ", ".join(f"{区域}:{峰值:.2f}" for 区域, 峰值 in 指标["PeakByRegion"].items())
        if 名称 == "基准折中":
            调度变化 = "基准"
            储能变化 = "基准"
        else:
            调度变化 = f"{任务变化数}个任务，主要{主要方向}"
            充电变化 = 储能策略["WeightedChargeHour"] - 基准储能["WeightedChargeHour"]
            放电变化 = 储能策略["WeightedDischargeHour"] - 基准储能["WeightedDischargeHour"]
            储能变化 = f"充电时刻{充电变化:+.2f}h，放电时刻{放电变化:+.2f}h"
        行 = {
            "情景": 名称,
            "运行成本_CNY": 指标["OperatingCost_CNY"],
            "碳排放_tCO2": 指标["CarbonEmission_tCO2"],
            "新能源利用率": 指标["RenewableUtilization"],
            "六区域峰值净购电功率_MW": 峰值向量,
            "任务调度变化": 调度变化,
            "储能策略变化": 储能变化,
            "调整任务数": 任务变化数,
            "平均网络时延_ms": 任务指标["AverageLatency_ms"],
            "P95网络时延_ms": 任务指标["P95Latency_ms"],
            "最大网络时延_ms": 任务指标["MaxLatency_ms"],
            "迁移率_%": 任务指标["MigrationRate"] * 100.0,
            "实时任务立即开工率_%": 任务指标["ImmediateStartRate"] * 100.0,
            "按时完成率_%": 任务指标["OnTimeCompletionRate"] * 100.0,
            "平均等待时间_h": 任务指标["AverageWaitingHour"],
            "P95等待时间_h": 任务指标["P95WaitingHour"],
            **{f"{区域}峰值_MW": 峰值 for 区域, 峰值 in 指标["PeakByRegion"].items()},
            **储能策略,
        }
        情景记录.append(行)
    return Pareto输出, 基准综合表, pd.DataFrame(情景记录)


def 检验任务约束(任务方案, 负荷数据, 情景名称="基准折中"):
    等待 = 任务方案["StartHour"] - np.maximum(任务方案["ArrivalHour"], 任务方案["EarliestStartHour"])
    时长误差 = np.abs(
        任务方案["FinishHour"] - 任务方案["StartHour"] - 任务方案["EstimatedDuration_min"] / 60.0
    )
    实时 = 任务方案["TaskType"] == "RealTimeInference"
    检验值 = {
        "任务未成功调度数": float((~任务方案["Scheduled"].astype(bool)).sum()),
        "任务重复数": float(任务方案["TaskID"].duplicated().sum()),
        "提前开工任务数": float((等待 < -1e-9).sum()),
        "连续执行时长最大误差_h": float(时长误差.max()),
        "实时任务未立即开工数": float((实时 & (np.abs(任务方案["StartHour"] - 任务方案["ArrivalHour"]) > 1e-9)).sum()),
        "网络时延超限任务数": float((任务方案["NetworkLatency_ms"] > 任务方案["MaxLatency_ms"] + 1e-9).sum()),
        "截止时间超限任务数": float(
            (
                (任务方案["FinishHour"] > 任务方案["LatestFinishHour"] + 1e-9)
                | (任务方案["FinishHour"] > 2406 + 1e-9)
            ).sum()
        ),
        "GPU容量超限时段数": float((负荷数据["GPU_Utilization"] > 1 + 1e-9).sum()),
        "IT功率超限时段数": float((负荷数据["IT_Utilization"] > 1 + 1e-9).sum()),
        "设施功率超限时段数": float((负荷数据["Facility_Utilization"] > 1 + 1e-9).sum()),
        "第2406小时AI负荷_MW": float(负荷数据.loc[负荷数据["Hour"] == 2406, "AI_IT_Load_MW"].abs().max()),
    }
    return pd.DataFrame(
        [
            {
                "Scenario": 情景名称,
                "Category": "任务",
                "Check": 名称,
                "Violation": 数值,
                "Passed": bool(数值 <= (1e-7 if "误差" in 名称 or "负荷" in 名称 else 0.0)),
            }
            for 名称, 数值 in 检验值.items()
        ]
    )


def 检验能源约束(能源结果, 储能信息, 情景名称, 碳上限=None, 弃电率上限=None):
    if 能源结果 is None:
        return pd.DataFrame(
            [{"Scenario": 情景名称, "Category": "能源", "Check": "模型可行性", "Violation": 1.0, "Passed": False}]
        )
    参数表 = 储能信息.set_index("Region")
    同时充放 = int(((能源结果["ChargePower_MW"] > 1e-5) & (能源结果["DischargePower_MW"] > 1e-5)).sum())
    同时购售 = int(((能源结果["GridPurchase_MW"] > 1e-5) & (能源结果["GridSell_MW"] > 1e-5)).sum())
    SOC下越界 = 0.0
    SOC上越界 = 0.0
    终端不足 = 0.0
    购电越界 = 0.0
    售电越界 = 0.0
    for 区域, 子表 in 能源结果.groupby("Region"):
        参数 = 参数表.loc[区域]
        SOC下越界 = max(SOC下越界, float(参数["MinSOC_MWh"] - 子表["SOC_MWh"].min()))
        SOC上越界 = max(SOC上越界, float(子表["SOC_MWh"].max() - 参数["StorageCapacity_MWh"]))
        终端不足 = max(终端不足, float(参数["InitialSOC_MWh"] - 子表.sort_values("Hour")["SOC_MWh"].iloc[-1]))
        购电越界 = max(购电越界, float(子表["GridPurchase_MW"].max() - 参数["MaxGridImport_MW"]))
        售电上限 = min(float(参数["SellLimit_MW"]), float(参数["MaxGridExport_MW"]))
        售电越界 = max(售电越界, float(子表["GridSell_MW"].max() - 售电上限))
    指标 = 计算能源指标(能源结果)
    检验值 = {
        "电力平衡最大残差_MW": float(能源结果["PowerBalanceResidual_MW"].abs().max()),
        "同时充放电时段数": float(同时充放),
        "同时购售电时段数": float(同时购售),
        "SOC下界超限_MWh": max(SOC下越界, 0.0),
        "SOC上界超限_MWh": max(SOC上越界, 0.0),
        "终端SOC不足_MWh": max(终端不足, 0.0),
        "购电功率超限_MW": max(购电越界, 0.0),
        "售电功率超限_MW": max(售电越界, 0.0),
        "碳约束超限_tCO2": max(指标["CarbonEmission_tCO2"] - float(碳上限), 0.0) if 碳上限 is not None else 0.0,
        "弃电率约束超限": max(指标["CurtailmentRate"] - float(弃电率上限), 0.0) if 弃电率上限 is not None else 0.0,
    }
    return pd.DataFrame(
        [
            {
                "Scenario": 情景名称,
                "Category": "能源",
                "Check": 名称,
                "Violation": 数值,
                "Passed": bool(数值 <= (1e-5 if "残差" in 名称 or "超限" in 名称 or "不足" in 名称 else 0.0)),
            }
            for 名称, 数值 in 检验值.items()
        ]
    )


def 配置绘图():
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#AAB7C4",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#DCE4EA",
            "grid.alpha": 0.7,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def 保存图形(图形, 图片目录, 文件名):
    图形.savefig(图片目录 / f"{文件名}.png", dpi=300, bbox_inches="tight")
    图形.savefig(图片目录 / f"{文件名}.pdf", bbox_inches="tight")
    plt.close(图形)


def 绘制问题四图表(Pareto表, 基准能源, 情景表, 情景任务映射, 图片目录):
    配置绘图()
    图片目录.mkdir(parents=True, exist_ok=True)
    if "运行成本_CNY" in Pareto表.columns:
        Pareto表 = Pareto表.rename(
            columns={
                "运行成本_CNY": "OperatingCost_CNY",
                "碳排放_tCO2": "CarbonEmission_tCO2",
                "新能源利用率": "RenewableUtilization",
                "是否选中": "Selected",
            }
        )
    Pareto表["Selected"] = Pareto表["Selected"].astype(str).str.lower().eq("true")
    图形 = plt.figure(figsize=(8.2, 6.0))
    坐标轴 = 图形.add_subplot(111, projection="3d")
    坐标轴.scatter(
        Pareto表["OperatingCost_CNY"] / 1e6,
        Pareto表["CarbonEmission_tCO2"] / 1e3,
        Pareto表["RenewableUtilization"] * 100,
        c="#539DCC",
        s=42,
        depthshade=False,
    )
    选中 = Pareto表[Pareto表["Selected"]]
    坐标轴.scatter(
        选中["OperatingCost_CNY"] / 1e6,
        选中["CarbonEmission_tCO2"] / 1e3,
        选中["RenewableUtilization"] * 100,
        c="#CE4459",
        s=90,
        marker="*",
        label="基准折中方案",
        depthshade=False,
    )
    坐标轴.set_xlabel("运行成本/百万CNY", labelpad=8)
    坐标轴.set_ylabel("碳排放/千tCO2", labelpad=8)
    坐标轴.set_zlabel("")
    图形.text(0.94, 0.50, "新能源利用率/%", rotation=90, va="center")
    坐标轴.view_init(elev=24, azim=-58)
    坐标轴.legend(loc="upper right", fontsize=10)
    图形.subplots_adjust(left=0.02, right=0.88, bottom=0.05, top=0.98)
    保存图形(图形, 图片目录, "图1_Pareto前沿与折中方案")

    日内 = 基准能源.assign(日内小时=基准能源["Hour"] % 24).groupby("日内小时").agg(
        设施负荷=("Facility_Load_MW", "mean"),
        新能源=("AvailableRenewable_MW", "mean"),
        净购电=("NetGridImport_MW", "mean"),
        充电=("ChargePower_MW", "mean"),
        放电=("DischargePower_MW", "mean"),
        SOC=("SOC_MWh", "mean"),
    )
    图形, 坐标轴 = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    坐标轴[0].plot(日内.index, 日内["设施负荷"], color="#0B559F", label="设施负荷")
    坐标轴[0].plot(日内.index, 日内["新能源"], color="#539DCC", label="可用新能源")
    坐标轴[0].plot(日内.index, 日内["净购电"], color="#CE4459", label="净购电")
    坐标轴[0].set_ylabel("平均功率/MW")
    坐标轴[0].legend(ncol=3)
    坐标轴[1].plot(日内.index, 日内["充电"], color="#88BEDC", label="充电")
    坐标轴[1].plot(日内.index, -日内["放电"], color="#E595A4", label="放电")
    右轴 = 坐标轴[1].twinx()
    右轴.plot(日内.index, 日内["SOC"], color="#50616F", label="SOC")
    坐标轴[1].set_xlabel("日内小时/h")
    坐标轴[1].set_ylabel("平均储能功率/MW")
    右轴.set_ylabel("平均SOC/MWh")
    坐标轴[1].legend(loc="upper left", ncol=2)
    右轴.legend(loc="upper right")
    图形.tight_layout()
    保存图形(图形, 图片目录, "图2_基准折中方案日内协同运行")

    绘制情景核心指标图(情景表, 图片目录)

    区域列表 = sorted(基准能源["Region"].unique())
    情景名称 = list(情景任务映射)
    GPU时矩阵 = []
    峰值矩阵 = []
    for 名称 in 情景名称:
        任务 = 情景任务映射[名称]
        GPU时 = (任务["GPU_Demand"] * 任务["EstimatedDuration_min"] / 60.0).groupby(任务["ExecutionRegion"]).sum()
        GPU时矩阵.append([float(GPU时.get(区域, 0.0)) for 区域 in 区域列表])
        行 = 情景表[情景表["情景"] == 名称].iloc[0]
        峰值矩阵.append([float(行.get(f"{区域}峰值_MW", np.nan)) for 区域 in 区域列表])
    色图 = LinearSegmentedColormap.from_list("低饱和蓝", ["#F7FAFC", "#BAD6EA", "#539DCC", "#0B559F"])
    图形, 坐标轴 = plt.subplots(1, 2, figsize=(12.0, 5.5))
    for 轴, 矩阵, 标签 in zip(坐标轴, [GPU时矩阵, 峰值矩阵], ["任务量/GPUh", "峰值净购电功率/MW"]):
        图像 = 轴.imshow(np.ma.masked_invalid(np.asarray(矩阵, dtype=float)), aspect="auto", cmap=色图)
        轴.set_xticks(range(len(区域列表)), 区域列表, rotation=30)
        轴.set_yticks(range(len(情景名称)), 情景名称)
        图形.colorbar(图像, ax=轴, label=标签, fraction=0.046, pad=0.04)
    图形.tight_layout()
    保存图形(图形, 图片目录, "图4_情景任务分配与区域峰值变化")


def 绘制情景核心指标图(情景表, 图片目录):
    配置绘图()
    有效情景 = 情景表.dropna(subset=["运行成本_CNY", "碳排放_tCO2", "新能源利用率"]).copy()
    基准行 = 有效情景[有效情景["情景"] == "基准折中"].iloc[0]
    指标设置 = [
        (
            (有效情景["运行成本_CNY"] - float(基准行["运行成本_CNY"]))
            / abs(float(基准行["运行成本_CNY"]))
            * 100.0,
            "运行成本变化/%",
            "#539DCC",
        ),
        (有效情景["碳排放_tCO2"], "碳排放/tCO2", "#CE4459"),
        (
            (有效情景["新能源利用率"] - float(基准行["新能源利用率"])) * 100.0,
            "新能源利用率变化/百分点",
            "#88BEDC",
        ),
    ]
    图形, 坐标轴 = plt.subplots(1, 3, figsize=(13.0, 4.8), sharey=False)
    for 轴, (数值, 标签, 颜色) in zip(坐标轴, 指标设置):
        轴.barh(有效情景["情景"], 数值, color=颜色)
        轴.axvline(0.0, color="#50616F", linewidth=0.9)
        轴.set_xlabel(标签)
    图形.tight_layout()
    保存图形(图形, 图片目录, "图3_单因素情景核心指标对比")


def 恢复情景任务映射(基准任务, 情景表, 迁移明细):
    映射 = {"基准折中": 基准任务.copy()}
    for 名称 in 情景表["情景"]:
        if 名称 == "基准折中":
            continue
        任务 = 基准任务.copy().set_index("TaskID")
        调整 = 迁移明细[迁移明细["Scenario"] == 名称].sort_values("Iteration") if len(迁移明细) else 迁移明细
        for 行 in 调整.itertuples(index=False):
            if 行.TaskID in 任务.index:
                任务.at[行.TaskID, "ExecutionRegion"] = 行.NewRegion
                任务.at[行.TaskID, "StartHour"] = float(行.NewStartHour)
                任务.at[行.TaskID, "FinishHour"] = float(行.NewStartHour) + float(任务.at[行.TaskID, "EstimatedDuration_min"]) / 60.0
        映射[名称] = 任务.reset_index()
    return 映射


def 转Markdown表(数据表, 小数位=4):
    列名 = list(数据表.columns)
    行文本 = ["| " + " | ".join(列名) + " |", "| " + " | ".join(["---"] * len(列名)) + " |"]
    for _, 行 in 数据表.iterrows():
        值列表 = []
        for 值 in 行:
            if isinstance(值, (float, np.floating)):
                值列表.append("" if pd.isna(值) else f"{值:.{小数位}f}")
            else:
                值列表.append(str(值))
        行文本.append("| " + " | ".join(值列表) + " |")
    return "\n".join(行文本)


def 生成问题四报告(Pareto表, 基准综合表, 情景表, 检验结果, 运行时间_s, 报告路径):
    Pareto展示 = Pareto表[
        [
            "方案",
            "协调前运行成本_CNY",
            "运行成本_CNY",
            "碳排放_tCO2",
            "新能源利用率",
            "平均网络时延_ms",
            "按时完成率",
            "最大区域峰值_MW",
            "协调后非支配",
            "理想点距离",
            "是否选中",
        ]
    ].rename(
        columns={
            "协调前运行成本_CNY": "协调前成本/CNY",
            "运行成本_CNY": "运行成本/CNY",
            "碳排放_tCO2": "碳排放/tCO2",
            "平均网络时延_ms": "平均网络时延/ms",
            "最大区域峰值_MW": "最大区域峰值/MW",
        }
    )
    情景展示 = 情景表[
        [
            "情景",
            "运行成本_CNY",
            "碳排放_tCO2",
            "新能源利用率",
            "平均网络时延_ms",
            "P95网络时延_ms",
            "按时完成率_%",
            "平均等待时间_h",
            "六区域峰值净购电功率_MW",
            "任务调度变化",
            "储能策略变化",
        ]
    ].rename(
        columns={
            "运行成本_CNY": "运行成本/CNY",
            "碳排放_tCO2": "碳排放/tCO2",
            "平均网络时延_ms": "平均网络时延/ms",
            "P95网络时延_ms": "P95网络时延/ms",
            "按时完成率_%": "按时完成率/%",
            "平均等待时间_h": "平均等待时间/h",
            "六区域峰值净购电功率_MW": "六区域峰值净购电功率/MW",
        }
    )
    通过数 = int(检验结果["Passed"].sum())
    内容 = f"""# 问题四计算结果

## 求解概况

采用三目标ε约束 Pareto、Gurobi 能源储能 MILP、固定整数状态 LP 条件影子价格和 EDF 优先成本感知列表调度。问题二任务方案作为初始解，问题三储能轨迹未直接复制。完整计算耗时 {运行时间_s:.2f} s。
任务数据使用第0—2399小时实际到达任务，第2400—2405小时仅用于结清末端任务，第2406小时不安排AI任务，仅保留固定负荷和储能结算。
固定任务负荷下得到的五个非支配点均从同一问题二任务方案出发，分别进行{统一协调迭代数}轮任务—能源联合协调；协调完成后重新筛选非支配解并选择理想点折中方案。基准与六个单因素情景同样采用统一起点和统一迭代轮数。

成本按购电支出减售电收入计算；净运行成本出现负值表示售电收入超过购电支出，不代表负的物理电价。高新能源波动情景若沿用基准碳上限不可行，则改用该情景的最小可行碳排放上限后再比较，结论表示场景内可行策略变化。

## Pareto方案及折中方案

{转Markdown表(Pareto展示)}

## 基准折中方案综合结果

{转Markdown表(基准综合表)}

## 单因素情景比较

{转Markdown表(情景展示)}

## 必要约束检验

共完成 {len(检验结果)} 项任务与能源约束检查，通过 {通过数} 项。详细数值见 `outputs/问题四计算结果/约束检验.csv`。

## 结果文件

- `Pareto方案及折中方案.csv`：五个初始非支配点的联合协调结果、协调后非支配标记和最终折中方案。
- `基准折中方案综合结果.csv`：成本、碳排放、新能源利用率、服务质量及六区域峰值。
- `单因素情景比较.csv`：七种单因素情景的结果与策略变化。
- `区域逐时协同运行.csv`、`最终任务调度方案.csv`：最终方案的逐时能源和逐任务结果。
- `迭代记录.csv`、`任务调整明细.csv`、`求解统计.csv`：协调过程与求解器审计记录。
"""
    报告路径.write_text(内容, encoding="utf-8")


def 合并并保存(表列表, 路径):
    非空表 = [表 for 表 in 表列表 if 表 is not None and len(表)]
    if 非空表:
        pd.concat(非空表, ignore_index=True).to_csv(路径, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(路径, index=False, encoding="utf-8-sig")


def 安全读取CSV(路径):
    try:
        return pd.read_csv(路径)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def 恢复已有基准(输出目录, 时间数据, GPU信息, 储能信息, 功率映射):
    基准任务 = pd.read_csv(输出目录 / "最终任务调度方案.csv")
    全部能源 = pd.read_csv(输出目录 / "区域逐时协同运行.csv")
    基准能源 = 全部能源[全部能源["Scenario"] == "基准折中"].copy().reset_index(drop=True)
    基准负荷 = 重构联合负荷(基准任务, 时间数据, GPU信息, 功率映射)
    Pareto输出 = pd.read_csv(输出目录 / "Pareto方案及折中方案.csv")
    Pareto表 = Pareto输出.rename(
        columns={
            "协调前运行成本_CNY": "InitialOperatingCost_CNY",
            "运行成本_CNY": "OperatingCost_CNY",
            "碳排放_tCO2": "CarbonEmission_tCO2",
            "新能源利用率": "RenewableUtilization",
            "平均网络时延_ms": "AverageLatency_ms",
            "P95网络时延_ms": "P95Latency_ms",
            "按时完成率": "OnTimeCompletionRate",
            "最大区域峰值_MW": "MaxRegionalPeak_MW",
            "协调后非支配": "CoordinatedNonDominated",
            "理想点距离": "IdealPointDistance",
            "是否选中": "Selected",
        }
    )
    Pareto表["Selected"] = Pareto表["Selected"].astype(str).str.lower().eq("true")
    Pareto表["CoordinatedNonDominated"] = (
        Pareto表["CoordinatedNonDominated"].astype(str).str.lower().eq("true")
    )
    Pareto表["CurtailmentRate"] = 1.0 - Pareto表["RenewableUtilization"]
    统计 = pd.read_csv(输出目录 / "求解统计.csv")
    端点名称 = ["最低成本端点", "最低碳排放端点", "最高新能源利用率端点"]
    端点 = 统计[统计["方案"].isin(端点名称)].drop_duplicates("方案")
    参数文件 = 输出目录 / "模型参数.json"
    if 参数文件.exists():
        参数 = json.loads(参数文件.read_text(encoding="utf-8"))
        Pareto范围 = {
            "CarbonMin_tCO2": float(参数["CarbonMin_tCO2"]),
            "CarbonMax_tCO2": float(参数["CarbonMax_tCO2"]),
            "CurtailmentMinRate": float(参数["CurtailmentMinRate"]),
            "CurtailmentMaxRate": float(参数["CurtailmentMaxRate"]),
        }
        碳上限 = float(参数["碳上限_tCO2"])
        弃电上限 = float(参数["弃电率上限"])
    else:
        Pareto范围 = {
            "CarbonMin_tCO2": float(端点["CarbonEmission_tCO2"].min()),
            "CarbonMax_tCO2": float(端点["CarbonEmission_tCO2"].max()),
            "CurtailmentMinRate": float(端点["CurtailmentRate"].min()),
            "CurtailmentMaxRate": float(端点["CurtailmentRate"].max()),
        }
    选中名称 = str(Pareto表.loc[Pareto表["Selected"], "方案"].iloc[0])
    if not 参数文件.exists():
        选中行 = Pareto表[Pareto表["Selected"]].iloc[0]
        碳上限 = float(选中行["CarbonEmission_tCO2"]) + 1e-6
        弃电上限 = 1.0 - float(选中行["RenewableUtilization"]) + 1e-10
    基准迭代 = 安全读取CSV(输出目录 / "迭代记录.csv")
    基准迭代 = (
        基准迭代[基准迭代["Scenario"] == f"{选中名称}_联合协调"].copy()
        if len(基准迭代)
        else 基准迭代
    )
    基准迁移 = 安全读取CSV(输出目录 / "任务调整明细.csv")
    基准迁移 = (
        基准迁移[基准迁移["Scenario"] == f"{选中名称}_联合协调"].copy()
        if len(基准迁移)
        else 基准迁移
    )
    return (
        基准任务,
        基准负荷,
        基准能源,
        计算能源指标(基准能源),
        基准迭代,
        基准迁移,
        统计,
        Pareto表,
        Pareto范围,
        碳上限,
        弃电上限,
    )


def 主程序():
    参数解析 = argparse.ArgumentParser()
    参数解析.add_argument("--time-limit", type=float, default=180.0)
    参数解析.add_argument("--log", action="store_true")
    参数解析.add_argument("--skip-scenarios", action="store_true")
    参数解析.add_argument("--scenarios-only", action="store_true")
    参数 = 参数解析.parse_args()
    开始时间 = time.perf_counter()
    根目录 = 项目根目录()
    输出目录 = 根目录 / "outputs" / "问题四计算结果"
    图片目录 = 根目录 / "figures" / "问题四计算结果"
    报告目录 = 根目录 / "reports"
    输出目录.mkdir(parents=True, exist_ok=True)
    图片目录.mkdir(parents=True, exist_ok=True)
    任务方案, 时间数据, GPU信息, 储能信息, 网络延迟, 功率映射 = 读取问题四数据(根目录)
    既有耗时 = 0.0
    if 参数.scenarios_only:
        if 参数.skip_scenarios:
            raise ValueError("--scenarios-only不能与--skip-scenarios同时使用")
        (
            基准任务,
            基准负荷,
            基准能源,
            基准指标,
            基准迭代,
            基准迁移,
            Pareto统计,
            Pareto表,
            Pareto范围,
            碳上限,
            弃电上限,
        ) = 恢复已有基准(输出目录, 时间数据, GPU信息, 储能信息, 功率映射)
        Pareto协调结果映射 = {}
        Pareto协调迭代 = 基准迭代
        Pareto协调迁移 = 基准迁移
        Pareto协调统计 = pd.DataFrame()
        统计路径 = 输出目录 / "运行统计.json"
        if 统计路径.exists():
            历史统计 = json.loads(统计路径.read_text(encoding="utf-8"))
            既有耗时 = float(历史统计["总运行时间_s"])
            if int(历史统计.get("单因素情景数", 1)) > 1:
                既有耗时 -= float(历史统计.get("本次运行时间_s", 0.0))
    else:
        初始负荷 = 重构联合负荷(任务方案, 时间数据, GPU信息, 功率映射)
        初始Pareto表, _, 初始能源映射, 初始Pareto统计, Pareto范围 = 构造Pareto前沿(
            初始负荷,
            储能信息,
            输出日志=参数.log,
            时间限制_s=参数.time_limit,
        )
        (
            Pareto表,
            选中方案,
            Pareto协调结果映射,
            Pareto协调迭代,
            Pareto协调迁移,
            Pareto协调统计,
        ) = 联合协调Pareto方案(
            初始Pareto表,
            初始能源映射,
            任务方案,
            时间数据,
            GPU信息,
            储能信息,
            网络延迟,
            功率映射,
            输出日志=参数.log,
            时间限制_s=参数.time_limit,
        )
        选中名称 = str(选中方案["方案"])
        选中结果 = Pareto协调结果映射[选中名称]
        基准任务 = 选中结果["任务"]
        基准负荷 = 选中结果["负荷"]
        基准能源 = 选中结果["能源"]
        基准指标 = 选中结果["指标"]
        基准迭代 = 选中结果["迭代"]
        基准迁移 = 选中结果["迁移"]
        碳上限 = 选中结果["碳上限"]
        弃电上限 = 选中结果["弃电上限"]
        Pareto统计 = pd.concat([初始Pareto统计, Pareto协调统计], ignore_index=True)
    if 基准能源 is None:
        raise RuntimeError("基准折中方案未得到可行解")
    情景结果 = []
    if not 参数.skip_scenarios:
        情景结果 = 运行单因素情景(
            任务方案,
            时间数据,
            GPU信息,
            储能信息,
            网络延迟,
            功率映射,
            碳上限,
            弃电上限,
            Pareto范围,
            输出日志=参数.log,
            时间限制_s=参数.time_limit,
        )
    Pareto输出, 基准综合表, 情景表 = 构造核心结果表(Pareto表, 基准任务, 基准能源, 情景结果)
    检验列表 = []
    if Pareto协调结果映射:
        for 方案 in Pareto协调结果映射.values():
            名称 = f"{方案['名称']}_联合协调"
            检验列表.append(检验任务约束(方案["任务"], 方案["负荷"], 名称))
            检验列表.append(
                检验能源约束(方案["能源"], 储能信息, 名称, 方案["碳上限"], 方案["弃电上限"])
            )
    else:
        检验列表.extend(
            [
                检验任务约束(基准任务, 基准负荷, "基准折中"),
                检验能源约束(基准能源, 储能信息, "基准折中", 碳上限, 弃电上限),
            ]
        )
    for 情景 in 情景结果:
        检验列表.extend(
            [
                检验任务约束(情景["任务"], 情景["负荷"], 情景["名称"]),
                检验能源约束(情景["能源"], 储能信息, 情景["名称"], 情景["碳上限"], 情景["弃电上限"]),
            ]
        )
    检验结果 = pd.concat(检验列表, ignore_index=True)
    所有能源 = []
    基准能源输出 = 基准能源.copy()
    基准能源输出["Scenario"] = "基准折中"
    所有能源.append(基准能源输出)
    所有迭代 = [Pareto协调迭代]
    所有迁移 = [Pareto协调迁移]
    所有统计 = [Pareto统计]
    情景任务映射 = {"基准折中": 基准任务}
    for 情景 in 情景结果:
        if 情景["能源"] is not None:
            能源 = 情景["能源"].copy()
            能源["Scenario"] = 情景["名称"]
            所有能源.append(能源)
            情景任务映射[情景["名称"]] = 情景["任务"]
        所有迭代.append(情景["迭代"])
        所有迁移.append(情景["迁移"])
        所有统计.append(情景["统计"])
    Pareto输出.to_csv(输出目录 / "Pareto方案及折中方案.csv", index=False, encoding="utf-8-sig")
    基准综合表.to_csv(输出目录 / "基准折中方案综合结果.csv", index=False, encoding="utf-8-sig")
    情景表.to_csv(输出目录 / "单因素情景比较.csv", index=False, encoding="utf-8-sig")
    基准任务.to_csv(输出目录 / "最终任务调度方案.csv", index=False, encoding="utf-8-sig")
    pd.concat(所有能源, ignore_index=True).to_csv(输出目录 / "区域逐时协同运行.csv", index=False, encoding="utf-8-sig")
    合并并保存(所有迭代, 输出目录 / "迭代记录.csv")
    合并并保存(所有迁移, 输出目录 / "任务调整明细.csv")
    合并并保存(所有统计, 输出目录 / "求解统计.csv")
    检验结果.to_csv(输出目录 / "约束检验.csv", index=False, encoding="utf-8-sig")
    绘制问题四图表(Pareto输出, 基准能源, 情景表, 情景任务映射, 图片目录)
    本次耗时 = time.perf_counter() - 开始时间
    总耗时 = 既有耗时 + 本次耗时
    运行统计 = {
        "任务数": int(len(基准任务)),
        "Pareto联合协调方案数": int(len(Pareto表)),
        "Pareto协调后非支配方案数": int(Pareto表["CoordinatedNonDominated"].sum()),
        "基准协调迭代数": int(len(基准迭代)),
        "统一协调迭代上限": 统一协调迭代数,
        "单因素情景数": int(len(情景结果) + 1),
        "约束检验通过数": int(检验结果["Passed"].sum()),
        "约束检验总数": int(len(检验结果)),
        "本次运行时间_s": 本次耗时,
        "总运行时间_s": 总耗时,
        "Gurobi版本": ".".join(map(str, gp.gurobi.version())),
    }
    (输出目录 / "运行统计.json").write_text(json.dumps(运行统计, ensure_ascii=False, indent=2), encoding="utf-8")
    (输出目录 / "模型参数.json").write_text(
        json.dumps(
            {
                "选中Pareto方案": str(Pareto表.loc[Pareto表["Selected"], "方案"].iloc[0]),
                "碳上限_tCO2": 碳上限,
                "弃电率上限": 弃电上限,
                "统一协调迭代上限": 统一协调迭代数,
                **Pareto范围,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    生成问题四报告(
        Pareto输出,
        基准综合表,
        情景表,
        检验结果,
        总耗时,
        报告目录 / "问题四计算结果.md",
    )
    print(json.dumps(运行统计, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
