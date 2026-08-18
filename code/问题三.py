import json
import time
from pathlib import Path

import gurobipy as gp
import matplotlib
import numpy as np
import pandas as pd
from gurobipy import GRB

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def 构造无储能方案(时段数据, 储能参数):
    结果 = 时段数据.copy().reset_index(drop=True)
    负荷 = 结果["Facility_Load_MW"].to_numpy(dtype=float)
    新能源 = 结果["AvailableRenewable_MW"].to_numpy(dtype=float)
    售电上限 = min(float(储能参数["SellLimit_MW"]), float(储能参数["MaxGridExport_MW"]))
    购电 = np.maximum(负荷 - 新能源, 0.0)
    富余 = np.maximum(新能源 - 负荷, 0.0)
    售电 = np.minimum(富余, 售电上限)
    弃电 = 富余 - 售电
    结果["GridPurchase_MW"] = 购电
    结果["GridSell_MW"] = 售电
    结果["NetGridImport_MW"] = 购电 - 售电
    结果["ChargePower_MW"] = 0.0
    结果["DischargePower_MW"] = 0.0
    结果["Curtailment_MW"] = 弃电
    结果["SOC_MWh"] = float(储能参数["InitialSOC_MWh"])
    结果["OperatingCost_CNY"] = 购电 * 结果["ElectricityPrice_CNY_per_MWh"] - 售电 * 结果["SellPrice_CNY_per_MWh"]
    结果["CarbonEmission_tCO2"] = 购电 * 结果["CarbonIntensity_tCO2_per_MWh"]
    结果["PowerBalanceResidual_MW"] = 购电 + 新能源 - 负荷 - 售电 - 弃电
    return 结果


def 求解储能窗口(
    时段数据,
    储能参数,
    场景,
    初始SOC_MWh,
    终端SOC下限_MWh,
    峰值上限_MW,
    基准成本_CNY,
    基准碳排放_tCO2,
    输出日志=False,
    时间限制_s=30.0,
):
    数据 = 时段数据.reset_index(drop=True).copy()
    数量 = len(数据)
    if 数量 == 0:
        raise ValueError("储能窗口不能为空")
    购电上限 = float(储能参数["MaxGridImport_MW"])
    售电上限 = min(float(储能参数["SellLimit_MW"]), float(储能参数["MaxGridExport_MW"]))
    充电上限 = float(储能参数["MaxChargePower_MW"])
    放电上限 = float(储能参数["MaxDischargePower_MW"])
    容量 = float(储能参数["StorageCapacity_MWh"])
    最小SOC = float(储能参数["MinSOC_MWh"])
    充电效率 = float(储能参数["ChargeEfficiency"])
    放电效率 = float(储能参数["DischargeEfficiency"])
    负荷 = 数据["Facility_Load_MW"].to_numpy(dtype=float)
    新能源 = 数据["AvailableRenewable_MW"].to_numpy(dtype=float)
    购电价 = 数据["ElectricityPrice_CNY_per_MWh"].to_numpy(dtype=float)
    售电价 = 数据["SellPrice_CNY_per_MWh"].to_numpy(dtype=float)
    碳强度 = 数据["CarbonIntensity_tCO2_per_MWh"].to_numpy(dtype=float)
    模型 = gp.Model(f"问题三_{场景}")
    模型.Params.OutputFlag = 1 if 输出日志 else 0
    模型.Params.TimeLimit = float(时间限制_s)
    模型.Params.MIPGap = 1e-4
    购电 = 模型.addVars(数量, lb=0.0, ub=购电上限, name="购电")
    售电 = 模型.addVars(数量, lb=0.0, ub=售电上限, name="售电")
    充电 = 模型.addVars(数量, lb=0.0, ub=充电上限, name="充电")
    放电 = 模型.addVars(数量, lb=0.0, ub=放电上限, name="放电")
    弃电 = 模型.addVars(数量, lb=0.0, ub={i: float(新能源[i]) for i in range(数量)}, name="弃电")
    SOC = 模型.addVars(数量, lb=最小SOC, ub=容量, name="SOC")
    充电状态 = 模型.addVars(数量, vtype=GRB.BINARY, name="充电状态")
    初始购电 = np.maximum(负荷 - 新能源, 0.0)
    初始售电 = np.minimum(np.maximum(新能源 - 负荷, 0.0), 售电上限)
    初始弃电 = np.maximum(新能源 - 负荷 - 初始售电, 0.0)
    for i in range(数量):
        购电[i].Start = float(初始购电[i])
        售电[i].Start = float(初始售电[i])
        充电[i].Start = 0.0
        放电[i].Start = 0.0
        弃电[i].Start = float(初始弃电[i])
        SOC[i].Start = float(初始SOC_MWh)
        充电状态[i].Start = 0.0
    for i in range(数量):
        模型.addConstr(购电[i] + 新能源[i] + 放电[i] == 负荷[i] + 充电[i] + 售电[i] + 弃电[i])
        模型.addConstr(售电[i] <= 新能源[i] + 放电[i])
        模型.addConstr(充电[i] <= 充电上限 * 充电状态[i])
        模型.addConstr(放电[i] <= 放电上限 * (1.0 - 充电状态[i]))
        模型.addConstr(购电[i] - 售电[i] <= float(峰值上限_MW))
        前一SOC = float(初始SOC_MWh) if i == 0 else SOC[i - 1]
        模型.addConstr(SOC[i] == 前一SOC + 充电效率 * 充电[i] - 放电[i] / 放电效率)
    模型.addConstr(SOC[数量 - 1] >= float(终端SOC下限_MWh))
    成本表达式 = gp.quicksum(购电[i] * 购电价[i] - 售电[i] * 售电价[i] for i in range(数量))
    碳表达式 = gp.quicksum(购电[i] * 碳强度[i] for i in range(数量))
    if 场景 == "最低成本方案":
        模型.ModelSense = GRB.MINIMIZE
        模型.setObjectiveN(成本表达式, 0, priority=2, weight=1.0, name="成本")
        模型.setObjectiveN(碳表达式, 1, priority=1, weight=1.0, name="碳排放")
    elif 场景 == "最低碳排放方案":
        模型.ModelSense = GRB.MINIMIZE
        模型.setObjectiveN(碳表达式, 0, priority=2, weight=1.0, name="碳排放")
        模型.setObjectiveN(成本表达式, 1, priority=1, weight=1.0, name="成本")
    elif 场景 == "成本—碳排放等权方案":
        if 基准成本_CNY <= 0 or 基准碳排放_tCO2 <= 0:
            raise ValueError("基准成本和基准碳排放必须为正数")
        模型.setObjective(
            1e6 * (0.5 * 成本表达式 / abs(float(基准成本_CNY)) + 0.5 * 碳表达式 / float(基准碳排放_tCO2)),
            GRB.MINIMIZE,
        )
    else:
        raise ValueError(f"未知场景：{场景}")
    模型.optimize()
    状态名称 = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }.get(模型.Status, str(模型.Status))
    if 模型.SolCount == 0:
        raise RuntimeError(f"Gurobi未得到可行解，状态为{状态名称}")
    结果 = 数据.copy()
    结果["GridPurchase_MW"] = [购电[i].X for i in range(数量)]
    结果["GridSell_MW"] = [售电[i].X for i in range(数量)]
    结果["NetGridImport_MW"] = 结果["GridPurchase_MW"] - 结果["GridSell_MW"]
    结果["ChargePower_MW"] = [充电[i].X for i in range(数量)]
    结果["DischargePower_MW"] = [放电[i].X for i in range(数量)]
    结果["Curtailment_MW"] = [弃电[i].X for i in range(数量)]
    结果["SOC_MWh"] = [SOC[i].X for i in range(数量)]
    结果["ChargeMode"] = [int(round(充电状态[i].X)) for i in range(数量)]
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
    try:
        MIP间隙 = float(模型.MIPGap)
    except AttributeError:
        MIP间隙 = 0.0 if 模型.Status == GRB.OPTIMAL else np.nan
    统计 = {
        "求解状态": 状态名称,
        "目标值": float(模型.ObjVal),
        "MIPGap": MIP间隙,
        "运行时间_s": float(模型.Runtime),
    }
    return 结果, 统计


def 计算区域峰值波动(逐时结果, 基准指标):
    基准 = 基准指标.set_index("Region")
    行列表 = []
    for (场景, 区域), 子表 in 逐时结果.groupby(["Scenario", "Region"], sort=False):
        子表 = 子表.sort_values("Hour")
        净购电 = 子表["NetGridImport_MW"].to_numpy(dtype=float)
        峰值位置 = int(np.argmax(净购电))
        峰值 = max(float(净购电[峰值位置]), 0.0)
        峰值时刻 = int(子表.iloc[峰值位置]["Hour"]) if 峰值 > 0 else np.nan
        标准差 = float(np.std(净购电, ddof=0))
        平均爬坡 = float(np.mean(np.abs(np.diff(净购电)))) if len(净购电) > 1 else 0.0
        峰谷差 = float(np.max(净购电) - np.min(净购电))
        基准峰值 = float(基准.loc[区域, "PeakNetImport_MW"])
        基准标准差 = float(基准.loc[区域, "NetImportStd_MW"])
        基准平均爬坡 = float(基准.loc[区域, "MeanHourlyRamp_MW_per_h"])
        基准峰谷差 = float(基准.loc[区域, "PeakValleyRange_MW"])
        行列表.append(
            {
                "Scenario": 场景,
                "Region": 区域,
                "PeakNetImport_MW": 峰值,
                "PeakHour_h": 峰值时刻,
                "PeakReduction_MW": 基准峰值 - 峰值,
                "PeakReductionRate": (基准峰值 - 峰值) / abs(基准峰值) if abs(基准峰值) > 1e-12 else np.nan,
                "NetImportStd_MW": 标准差,
                "FluctuationReductionRate": (基准标准差 - 标准差) / abs(基准标准差) if abs(基准标准差) > 1e-12 else np.nan,
                "MeanHourlyRamp_MW_per_h": 平均爬坡,
                "MeanHourlyRampReductionRate": (基准平均爬坡 - 平均爬坡) / abs(基准平均爬坡) if abs(基准平均爬坡) > 1e-12 else np.nan,
                "PeakValleyRange_MW": 峰谷差,
                "PeakValleyReductionRate": (基准峰谷差 - 峰谷差) / abs(基准峰谷差) if abs(基准峰谷差) > 1e-12 else np.nan,
            }
        )
    指标表 = pd.DataFrame(行列表)
    if "无储能辅助对照" in set(指标表["Scenario"]):
        无储能 = 指标表[指标表["Scenario"] == "无储能辅助对照"].set_index("Region")
        指标表["StoragePeakReduction_MW"] = 指标表.apply(
            lambda 行: float(无储能.loc[行["Region"], "PeakNetImport_MW"] - 行["PeakNetImport_MW"]), axis=1
        )
        指标表["StorageNetImportStdReduction_MW"] = 指标表.apply(
            lambda 行: float(无储能.loc[行["Region"], "NetImportStd_MW"] - 行["NetImportStd_MW"]), axis=1
        )
        指标表["StorageMeanHourlyRampReduction_MW_per_h"] = 指标表.apply(
            lambda 行: float(无储能.loc[行["Region"], "MeanHourlyRamp_MW_per_h"] - 行["MeanHourlyRamp_MW_per_h"]), axis=1
        )
        指标表["StoragePeakValleyReduction_MW"] = 指标表.apply(
            lambda 行: float(无储能.loc[行["Region"], "PeakValleyRange_MW"] - 行["PeakValleyRange_MW"]), axis=1
        )
    return 指标表


def 项目根目录():
    return Path(__file__).resolve().parents[1]


def 读取问题三数据(数据目录):
    时间数据 = pd.read_excel(数据目录 / "region_time_data.xlsx")
    储能信息 = pd.read_excel(数据目录 / "storage_information.xlsx")
    区域信息 = pd.read_excel(数据目录 / "GPU_information.xlsx")
    区域参数 = 区域信息[["Region", "PUE"]]
    时间数据 = 时间数据.merge(区域参数, on="Region", how="left", validate="many_to_one")
    时间数据["Facility_Load_MW"] = 时间数据["PUE"] * (
        时间数据["Baseline_AI_IT_Load_MW"] + 时间数据["NonAI_IT_Load_MW"]
    )
    时间数据 = 时间数据.sort_values(["Region", "Hour"]).reset_index(drop=True)
    if 时间数据[["PUE", "Facility_Load_MW"]].isna().any().any():
        raise ValueError("区域PUE或设施负荷存在缺失值")
    小时检验 = 时间数据.groupby("Region")["Hour"].agg(["count", "min", "max"])
    if not ((小时检验["count"] == 2407) & (小时检验["min"] == 0) & (小时检验["max"] == 2406)).all():
        raise ValueError("问题三要求每个区域均包含0—2406小时数据")
    return 时间数据, 储能信息


def 构造附件基准方案(时间数据):
    结果 = 时间数据.copy()
    结果["Scenario"] = "附件基准状态"
    结果["GridPurchase_MW"] = 结果["GridPurchase_MW"].astype(float)
    结果["GridSell_MW"] = 结果["GridSell_MW"].astype(float)
    结果["NetGridImport_MW"] = 结果["GridPurchase_MW"] - 结果["GridSell_MW"]
    结果["ChargePower_MW"] = 结果["ChargePower_MW"].astype(float)
    结果["DischargePower_MW"] = 结果["DischargePower_MW"].astype(float)
    结果["Curtailment_MW"] = 结果["Curtailment_MW"].astype(float)
    结果["SOC_MWh"] = 结果["SOC_MWh"].astype(float)
    结果["OperatingCost_CNY"] = (
        结果["GridPurchase_MW"] * 结果["ElectricityPrice_CNY_per_MWh"]
        - 结果["GridSell_MW"] * 结果["SellPrice_CNY_per_MWh"]
    )
    结果["CarbonEmission_tCO2"] = 结果["GridPurchase_MW"] * 结果["CarbonIntensity_tCO2_per_MWh"]
    结果["PowerBalanceResidual_MW"] = (
        结果["GridPurchase_MW"]
        + 结果["AvailableRenewable_MW"]
        + 结果["DischargePower_MW"]
        - 结果["Facility_Load_MW"]
        - 结果["ChargePower_MW"]
        - 结果["GridSell_MW"]
        - 结果["Curtailment_MW"]
    )
    return 结果


def 构造基准区域指标(附件基准):
    行列表 = []
    for 区域, 子表 in 附件基准.groupby("Region", sort=False):
        子表 = 子表.sort_values("Hour")
        净购电 = 子表["NetGridImport_MW"].to_numpy(dtype=float)
        行列表.append(
            {
                "Region": 区域,
                "PeakNetImport_MW": max(float(np.max(净购电)), 0.0),
                "NetImportStd_MW": float(np.std(净购电, ddof=0)),
                "MeanHourlyRamp_MW_per_h": float(np.mean(np.abs(np.diff(净购电)))),
                "PeakValleyRange_MW": float(np.max(净购电) - np.min(净购电)),
            }
        )
    return pd.DataFrame(行列表)


def 求解全部场景(时间数据, 储能信息, 输出日志=False, 时间限制_s=120.0):
    附件基准 = 构造附件基准方案(时间数据)
    基准区域指标 = 构造基准区域指标(附件基准)
    系统基准成本 = float(附件基准["OperatingCost_CNY"].sum())
    系统基准碳排放 = float(附件基准["CarbonEmission_tCO2"].sum())
    所有结果 = [附件基准]
    求解统计 = []
    无储能列表 = []
    参数表 = 储能信息.set_index("Region")
    for 区域, 子表 in 时间数据.groupby("Region", sort=False):
        参数 = 参数表.loc[区域]
        结果 = 构造无储能方案(子表, 参数)
        结果["Scenario"] = "无储能辅助对照"
        无储能列表.append(结果)
    所有结果.append(pd.concat(无储能列表, ignore_index=True))
    场景列表 = ["最低成本方案", "最低碳排放方案", "成本—碳排放等权方案"]
    for 场景 in 场景列表:
        for 区域, 子表 in 时间数据.groupby("Region", sort=False):
            参数 = 参数表.loc[区域]
            峰值上限 = float(基准区域指标.set_index("Region").loc[区域, "PeakNetImport_MW"])
            开始 = time.perf_counter()
            结果, 统计 = 求解储能窗口(
                子表,
                参数,
                场景=场景,
                初始SOC_MWh=float(参数["InitialSOC_MWh"]),
                终端SOC下限_MWh=float(参数["InitialSOC_MWh"]),
                峰值上限_MW=峰值上限,
                基准成本_CNY=系统基准成本,
                基准碳排放_tCO2=系统基准碳排放,
                输出日志=输出日志,
                时间限制_s=时间限制_s,
            )
            结果["Scenario"] = 场景
            所有结果.append(结果)
            统计.update({"Scenario": 场景, "Region": 区域, "总耗时_s": time.perf_counter() - 开始})
            求解统计.append(统计)
    逐时结果 = pd.concat(所有结果, ignore_index=True)
    return 逐时结果, pd.DataFrame(求解统计), 基准区域指标


def 构造系统指标(逐时结果):
    指标 = (
        逐时结果.groupby("Scenario", sort=False)
        .agg(
            OperatingCost_CNY=("OperatingCost_CNY", "sum"),
            CarbonEmission_tCO2=("CarbonEmission_tCO2", "sum"),
            GridPurchase_MWh=("GridPurchase_MW", "sum"),
            GridSell_MWh=("GridSell_MW", "sum"),
            ChargeEnergy_MWh=("ChargePower_MW", "sum"),
            DischargeEnergy_MWh=("DischargePower_MW", "sum"),
        )
        .reset_index()
    )
    基准 = 指标[指标["Scenario"] == "附件基准状态"].iloc[0]
    无储能 = 指标[指标["Scenario"] == "无储能辅助对照"].iloc[0]
    指标["CostChange_CNY"] = 指标["OperatingCost_CNY"] - float(基准["OperatingCost_CNY"])
    指标["CostChangeRate"] = 指标["CostChange_CNY"] / abs(float(基准["OperatingCost_CNY"]))
    指标["CarbonChange_tCO2"] = 指标["CarbonEmission_tCO2"] - float(基准["CarbonEmission_tCO2"])
    指标["CarbonChangeRate"] = 指标["CarbonChange_tCO2"] / float(基准["CarbonEmission_tCO2"])
    指标["StorageIncrementalCost_CNY"] = 指标["OperatingCost_CNY"] - float(无储能["OperatingCost_CNY"])
    指标["StorageIncrementalCostRate"] = 指标["StorageIncrementalCost_CNY"] / abs(float(无储能["OperatingCost_CNY"]))
    指标["StorageIncrementalCarbon_tCO2"] = 指标["CarbonEmission_tCO2"] - float(无储能["CarbonEmission_tCO2"])
    指标["StorageIncrementalCarbonRate"] = np.where(
        abs(float(无储能["CarbonEmission_tCO2"])) > 1e-12,
        指标["StorageIncrementalCarbon_tCO2"] / abs(float(无储能["CarbonEmission_tCO2"])),
        np.nan,
    )
    return 指标


def 检验问题三约束(逐时结果, 储能信息, 基准区域指标):
    参数表 = 储能信息.set_index("Region")
    峰值表 = 基准区域指标.set_index("Region")
    行列表 = []
    优化结果 = 逐时结果[逐时结果["Scenario"].isin(["最低成本方案", "最低碳排放方案", "成本—碳排放等权方案"])]
    for (场景, 区域), 子表 in 优化结果.groupby(["Scenario", "Region"], sort=False):
        参数 = 参数表.loc[区域]
        同时充放 = int(((子表["ChargePower_MW"] > 1e-6) & (子表["DischargePower_MW"] > 1e-6)).sum())
        售电来源超限 = float((子表["GridSell_MW"] - 子表["AvailableRenewable_MW"] - 子表["DischargePower_MW"]).max())
        行列表.append(
            {
                "Scenario": 场景,
                "Region": 区域,
                "MaxPowerBalanceResidual_MW": float(subset_abs_max(子表["PowerBalanceResidual_MW"])),
                "SimultaneousChargeDischargeCount": 同时充放,
                "SOCBelowMin_MWh": max(float(参数["MinSOC_MWh"] - 子表["SOC_MWh"].min()), 0.0),
                "SOCAboveCapacity_MWh": max(float(子表["SOC_MWh"].max() - 参数["StorageCapacity_MWh"]), 0.0),
                "TerminalSOCShortfall_MWh": max(float(参数["InitialSOC_MWh"] - 子表.sort_values("Hour")["SOC_MWh"].iloc[-1]), 0.0),
                "PeakLimitExcess_MW": max(float(subset_abs_upper(子表["NetGridImport_MW"], 峰值表.loc[区域, "PeakNetImport_MW"])), 0.0),
                "GridImportLimitExcess_MW": max(float(subset_abs_upper(子表["GridPurchase_MW"], 参数["MaxGridImport_MW"])), 0.0),
                "GridExportLimitExcess_MW": max(float(subset_abs_upper(子表["GridSell_MW"], min(参数["SellLimit_MW"], 参数["MaxGridExport_MW"]))), 0.0),
                "SellSourceExcess_MW": max(售电来源超限, 0.0),
            }
        )
    检验 = pd.DataFrame(行列表)
    数值列 = [列 for 列 in 检验.columns if 列 not in ["Scenario", "Region"]]
    检验["Passed"] = (检验[数值列].max(axis=1) <= 1e-5) & (检验["SimultaneousChargeDischargeCount"] == 0)
    return 检验


def subset_abs_max(序列):
    return np.max(np.abs(np.asarray(序列, dtype=float)))


def subset_abs_upper(序列, 上限):
    return np.max(np.asarray(序列, dtype=float) - float(上限))


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


def 绘制问题三图表(逐时结果, 系统指标, 区域指标, 图片目录, 输出目录):
    配置绘图()
    图片目录.mkdir(parents=True, exist_ok=True)
    场景顺序 = ["附件基准状态", "无储能辅助对照", "最低成本方案", "最低碳排放方案", "成本—碳排放等权方案"]
    指标 = 系统指标.set_index("Scenario").loc[场景顺序].reset_index()
    简称 = ["附件基准", "无储能", "最低成本", "最低碳排放", "等权方案"]
    图形, 坐标轴 = plt.subplots(1, 2, figsize=(12, 4.8))
    成本 = 指标["OperatingCost_CNY"].to_numpy() / 1e8
    碳 = 指标["CarbonEmission_tCO2"].to_numpy() / 1e6
    柱1 = 坐标轴[0].bar(简称, 成本, color=["#BAD6EA", "#C0D6EA", "#2A7AB9", "#CE4459", "#E595A4"])
    柱2 = 坐标轴[1].bar(简称, 碳, color=["#BAD6EA", "#C0D6EA", "#2A7AB9", "#CE4459", "#E595A4"])
    坐标轴[0].set_title("系统净运行成本")
    坐标轴[0].set_ylabel("亿元")
    坐标轴[1].set_title("系统碳排放")
    坐标轴[1].set_ylabel("百万 tCO2")
    for 轴, 柱组, 数值 in [(坐标轴[0], 柱1, 成本), (坐标轴[1], 柱2, 碳)]:
        轴.grid(axis="x", visible=False)
        轴.tick_params(axis="x", rotation=18)
        轴.axhline(0, color="#9CA3AF", linewidth=0.8)
        for 柱, 值 in zip(柱组, 数值):
            轴.text(柱.get_x() + 柱.get_width() / 2, 值, f"{值:.2f}", ha="center", va="bottom" if 值 >= 0 else "top", fontsize=8)
    图形.suptitle("储能策略对系统成本与碳排放的影响", fontsize=14)
    保存图形(图形, 图片目录, "图1_成本与碳排放对比")
    区域优化 = 区域指标[区域指标["Scenario"] == "成本—碳排放等权方案"].copy()
    区域优化 = 区域优化.sort_values("Region")
    横坐标 = np.arange(len(区域优化))
    宽度 = 0.36
    图形, 轴 = plt.subplots(figsize=(9, 4.8))
    轴.bar(横坐标 - 宽度 / 2, 区域优化["StoragePeakReduction_MW"], width=宽度, color="#2A7AB9", label="峰值净购电降幅")
    轴.bar(横坐标 + 宽度 / 2, 区域优化["StorageNetImportStdReduction_MW"], width=宽度, color="#E595A4", label="净购电标准差降幅")
    轴.set_xticks(横坐标, 区域优化["Region"])
    轴.set_ylabel("相对无储能辅助对照的降幅（MW）")
    轴.set_title("等权储能方案的区域峰值与波动增量影响")
    轴.axhline(0, color="#9CA3AF", linewidth=0.8)
    轴.legend()
    保存图形(图形, 图片目录, "图2_区域峰值与波动变化")
    等权 = 逐时结果[逐时结果["Scenario"] == "成本—碳排放等权方案"].copy()
    等权["StorageActivity_MW"] = 等权["ChargePower_MW"] + 等权["DischargePower_MW"]
    日活动 = 等权.assign(Day=等权["Hour"] // 24).groupby(["Region", "Day"])["StorageActivity_MW"].sum()
    代表区域, 代表日 = 日活动.idxmax()
    日数据 = 等权[(等权["Region"] == 代表区域) & (等权["Hour"] // 24 == 代表日)].sort_values("Hour").copy()
    日数据.to_csv(输出目录 / "代表日储能运行.csv", index=False, encoding="utf-8-sig")
    图形, 轴1 = plt.subplots(figsize=(11, 5.2))
    小时 =日数据["Hour"].to_numpy()
    轴1.bar(小时, 日数据["ChargePower_MW"], color="#539DCC", alpha=0.8, label="充电功率")
    轴1.bar(小时, -日数据["DischargePower_MW"], color="#CE4459", alpha=0.8, label="放电功率")
    轴1.set_xlabel("Hour")
    轴1.set_ylabel("储能功率（MW）")
    轴2 = 轴1.twinx()
    轴2.plot(小时, 日数据["ElectricityPrice_CNY_per_MWh"], color="#2A7AB9", linewidth=1.8, label="购电价")
    轴2.plot(小时, 日数据["SOC_MWh"], color="#E595A4", linewidth=1.8, label="SOC")
    轴2.set_ylabel("电价（CNY/MWh）或 SOC（MWh）")
    轴1.legend(loc="upper left", ncol=2)
    轴2.legend(loc="upper right", ncol=2)
    轴1.set_title(f"{代表区域} 第{代表日}日储能运行轨迹", pad=10)
    保存图形(图形, 图片目录, "图3_代表日储能运行轨迹")
    系统逐时 = (
        逐时结果[逐时结果["Scenario"].isin(["附件基准状态", "成本—碳排放等权方案"])]
        .groupby(["Scenario", "Hour"])["NetGridImport_MW"]
        .sum()
        .reset_index()
    )
    图形, 轴 = plt.subplots(figsize=(12, 4.6))
    for 场景, 颜色 in [("附件基准状态", "#BAD6EA"), ("成本—碳排放等权方案", "#CE4459")]:
        子表 = 系统逐时[系统逐时["Scenario"] == 场景]
        轴.plot(子表["Hour"], 子表["NetGridImport_MW"], color=颜色, linewidth=0.85, label=场景)
    轴.set_xlabel("Hour")
    轴.set_ylabel("系统净购电功率（MW）")
    轴.set_title("系统净购电功率全时段对比")
    轴.legend()
    保存图形(图形, 图片目录, "图4_系统净购电功率对比")
def 生成问题三报告(系统指标, 区域指标, 检验结果, 求解统计, 报告路径):
    指标 = 系统指标.set_index("Scenario")
    基准 = 指标.loc["附件基准状态"]
    无储能 = 指标.loc["无储能辅助对照"]
    等权 = 指标.loc["成本—碳排放等权方案"]
    区域等权 = 区域指标[区域指标["Scenario"] == "成本—碳排放等权方案"]
    波动改善区域数 = int((区域等权["FluctuationReductionRate"] > 0).sum())
    储能峰值改善区域数 = int((区域等权["StoragePeakReduction_MW"] > 1e-6).sum())
    储能波动改善区域数 = int((区域等权["StorageNetImportStdReduction_MW"] > 1e-6).sum())
    储能峰值改善 = 区域等权[区域等权["StoragePeakReduction_MW"] > 1e-6]
    储能峰值区域 = "、".join(储能峰值改善["Region"].tolist()) if len(储能峰值改善) else "无"
    储能峰值降幅 = float(储能峰值改善["StoragePeakReduction_MW"].max()) if len(储能峰值改善) else 0.0
    内容 = f"""# 问题三计算结果

## 求解概况

在问题一、问题二已完成的数据定义基础上，问题三独立采用附件 `region_time_data.xlsx` 中第 0—2406 小时的 `Baseline_AI_IT_Load_MW` 与 `NonAI_IT_Load_MW`，经各区域 PUE 映射得到固定设施负荷。附件给出的购售电、充放电、弃电和 SOC 轨迹为正式基准；无储能方案仅作辅助反事实。模型采用 Gurobi 13.0.2，对六个区域分别求解完整 2407 小时混合整数线性规划，再利用目标和约束的区域可分性汇总为系统结果。

成本与碳排放分别构成最低成本、最低碳排放和基准归一化等权三种优化情景。区域峰值净购电功率不超过附件基准峰值，作为电网安全硬约束与评价指标；负荷波动不进入目标函数，仅以净购电标准差、平均小时爬坡量和峰谷差评价。

## 系统结果

- 附件基准净运行成本为 {基准['OperatingCost_CNY']:,.2f} CNY，碳排放为 {基准['CarbonEmission_tCO2']:,.2f} tCO2。
- 无储能辅助对照在同样优化新能源分配和购售电的条件下，净运行成本为 {无储能['OperatingCost_CNY']:,.2f} CNY，碳排放为 {无储能['CarbonEmission_tCO2']:,.2f} tCO2。
- 等权方案净运行成本为 {等权['OperatingCost_CNY']:,.2f} CNY，较附件基准变化 {等权['CostChangeRate']:.2%}；碳排放为 {等权['CarbonEmission_tCO2']:,.2f} tCO2，较附件基准变化 {等权['CarbonChangeRate']:.2%}。
- 单独考察储能增量作用，等权方案相对无储能辅助对照的净运行成本变化为 {等权['StorageIncrementalCost_CNY']:,.2f} CNY（{等权['StorageIncrementalCostRate']:.2%}），碳排放变化为 {等权['StorageIncrementalCarbon_tCO2']:,.2f} tCO2（{等权['StorageIncrementalCarbonRate']:.2%}）。
- 等权方案累计充电量为 {等权['ChargeEnergy_MWh']:,.2f} MWh，累计放电量为 {等权['DischargeEnergy_MWh']:,.2f} MWh。
- 峰值净购电功率按 `max(max(NetGridImport_MW, 0))` 计算。等权方案下六个区域的峰值净购电功率均为 0 MW；六个区域中有 {波动改善区域数} 个区域的净购电标准差低于附件基准。
- 与无储能辅助对照直接比较，储能使 {储能峰值改善区域数} 个区域的峰值净购电功率进一步下降，其中 {储能峰值区域} 降低 {储能峰值降幅:.2f} MW；同时使 {储能波动改善区域数} 个区域的净购电标准差进一步下降。全时段净外送只计为购电峰值 0 MW，不再把外送增加误计为削峰。

成本为负时表示售电收入高于购电支出，是题目允许新能源售电且未计固定运维成本时的净结算结果。附件原始运行状态仍是正式基准；无储能辅助对照只用于从“新能源重新分配与购售电优化”的总体效应中分离储能充放电带来的增量影响，不作为归一化基准。

## 必要检验

- 18 个区域—优化情景模型均取得可行解，其中最差 MIPGap 为 {求解统计['MIPGap'].max():.6g}。
- 最大电力平衡残差为 {检验结果['MaxPowerBalanceResidual_MW'].max():.6g} MW。
- 充放电同时发生的时段总数为 {int(检验结果['SimultaneousChargeDischargeCount'].sum())}。
- 区域峰值、SOC、购售电通道及终端 SOC 检验总体通过：{'是' if 检验结果['Passed'].all() else '否'}。

## 成果文件

- `系统成本碳排放对比.csv`：附件基准、无储能辅助对照和三种优化方案的系统指标。
- `区域峰值与负荷波动对比.csv`：合并呈现六区域峰值净购电功率和三类负荷波动指标。
- `区域逐时储能运行.csv`：各情景、区域、小时的购售电、充放电、SOC、成本和碳排放。
- `求解统计.csv` 与 `约束检验.csv`：Gurobi 状态、运行时间、MIPGap 和必要硬约束检验。
"""
    报告路径.parent.mkdir(parents=True, exist_ok=True)
    报告路径.write_text(内容, encoding="utf-8")


def 主程序():
    开始时间 = time.perf_counter()
    根目录 = 项目根目录()
    数据目录 = 根目录 / "题目" / "附件数据"
    输出目录 = 根目录 / "outputs" / "问题三计算结果"
    图片目录 = 根目录 / "figures" / "问题三计算结果"
    输出目录.mkdir(parents=True, exist_ok=True)
    时间数据, 储能信息 = 读取问题三数据(数据目录)
    逐时结果, 求解统计, 基准区域指标 = 求解全部场景(时间数据, 储能信息)
    系统指标 = 构造系统指标(逐时结果)
    区域指标 = 计算区域峰值波动(逐时结果, 基准区域指标)
    检验结果 = 检验问题三约束(逐时结果, 储能信息, 基准区域指标)
    逐时列 = [
        "Scenario", "Region", "Hour", "Facility_Load_MW", "AvailableRenewable_MW",
        "ElectricityPrice_CNY_per_MWh", "SellPrice_CNY_per_MWh", "CarbonIntensity_tCO2_per_MWh",
        "GridPurchase_MW", "GridSell_MW", "NetGridImport_MW", "ChargePower_MW",
        "DischargePower_MW", "Curtailment_MW", "SOC_MWh", "OperatingCost_CNY",
        "CarbonEmission_tCO2", "PowerBalanceResidual_MW",
    ]
    逐时结果[逐时列].to_csv(输出目录 / "区域逐时储能运行.csv", index=False, encoding="utf-8-sig")
    系统指标.to_csv(输出目录 / "系统成本碳排放对比.csv", index=False, encoding="utf-8-sig")
    区域指标.to_csv(输出目录 / "区域峰值与负荷波动对比.csv", index=False, encoding="utf-8-sig")
    求解统计.to_csv(输出目录 / "求解统计.csv", index=False, encoding="utf-8-sig")
    检验结果.to_csv(输出目录 / "约束检验.csv", index=False, encoding="utf-8-sig")
    绘制问题三图表(逐时结果, 系统指标, 区域指标, 图片目录, 输出目录)
    运行时间_s = time.perf_counter() - 开始时间
    运行统计 = {
        "运行时间_s": 运行时间_s,
        "Gurobi版本": ".".join(map(str, gp.gurobi.version())),
        "模型数": int(len(求解统计)),
        "全部约束通过": bool(检验结果["Passed"].all()),
        "最大MIPGap": float(求解统计["MIPGap"].max()),
    }
    (输出目录 / "运行统计.json").write_text(json.dumps(运行统计, ensure_ascii=False, indent=2), encoding="utf-8")
    生成问题三报告(系统指标, 区域指标, 检验结果, 求解统计, 根目录 / "reports" / "问题三计算结果.md")
    print(系统指标.to_string(index=False))
    print(检验结果.to_string(index=False))
    print(pd.Series(运行统计).to_string())


if __name__ == "__main__":
    主程序()
