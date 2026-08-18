import importlib.util
import io
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd


模块路径 = Path(__file__).resolve().parents[1] / "code" / "问题四.py"
模块规格 = importlib.util.spec_from_file_location("问题四", 模块路径)
问题四 = importlib.util.module_from_spec(模块规格)
模块规格.loader.exec_module(问题四)


def test_小时重叠与峰值净购电口径():
    assert 问题四.计算小时重叠(1.5, 3.25, 1) == 0.5
    assert 问题四.计算小时重叠(1.5, 3.25, 2) == 1.0
    数据 = pd.DataFrame(
        {
            "Region": ["A", "A", "B", "B"],
            "AvailableRenewable_MW": [1.0, 1.0, 1.0, 1.0],
            "Curtailment_MW": [0.0, 0.0, 0.0, 0.0],
            "NetGridImport_MW": [-5.0, -2.0, 3.0, -1.0],
            "OperatingCost_CNY": [0.0] * 4,
            "CarbonEmission_tCO2": [0.0] * 4,
        }
    )
    指标 = 问题四.计算能源指标(数据)
    assert 指标["PeakByRegion"]["A"] == 0.0
    assert 指标["PeakByRegion"]["B"] == 3.0


def test_非支配筛选与理想点选择():
    方案 = pd.DataFrame(
        {
            "方案": ["甲", "乙", "丙", "丁"],
            "OperatingCost_CNY": [1.0, 2.0, 1.5, 3.0],
            "CarbonEmission_tCO2": [3.0, 1.0, 2.0, 3.0],
            "CurtailmentRate": [3.0, 3.0, 1.0, 4.0],
        }
    )
    非支配 = 问题四.删除支配解(方案)
    assert set(非支配["方案"]) == {"甲", "乙", "丙"}
    结果, 选中 = 问题四.选择理想点方案(非支配)
    assert 结果["Selected"].sum() == 1
    assert 选中["方案"] == "丙"


def test_单因素变换只修改指定字段():
    数据 = pd.DataFrame(
        {
            "Region": ["A"] * 4,
            "Hour": range(4),
            "ElectricityPrice_CNY_per_MWh": [1.0, 2.0, 3.0, 4.0],
            "AvailableRenewable_MW": [1.0, 4.0, 1.0, 4.0],
            "SellPrice_CNY_per_MWh": [0.5] * 4,
        }
    )
    平价 = 问题四.构造电价情景(数据, 0.0)
    assert np.allclose(平价["ElectricityPrice_CNY_per_MWh"], 2.5)
    assert np.allclose(平价["AvailableRenewable_MW"], 数据["AvailableRenewable_MW"])
    低波动 = 问题四.构造新能源波动情景(数据, 0.5)
    assert 低波动["AvailableRenewable_MW"].std() < 数据["AvailableRenewable_MW"].std()
    assert np.allclose(低波动["ElectricityPrice_CNY_per_MWh"], 数据["ElectricityPrice_CNY_per_MWh"])


def test_实际任务负荷重构满足边界():
    根目录 = Path(__file__).resolve().parents[1]
    任务, 时间, GPU信息, _, _, 功率映射 = 问题四.读取问题四数据(根目录)
    负荷 = 问题四.重构联合负荷(任务, 时间, GPU信息, 功率映射)
    assert len(负荷) == 6 * 2407
    assert 负荷.loc[负荷["Hour"] == 2406, "AI_IT_Load_MW"].abs().max() <= 1e-12
    assert 负荷[["GPU_Utilization", "IT_Utilization", "Facility_Utilization"]].max().max() <= 1 + 1e-9


def test_代码无注释且无用户名绝对路径():
    源码 = 模块路径.read_text(encoding="utf-8")
    标记 = tokenize.generate_tokens(io.StringIO(源码).readline)
    assert not any(类型 == tokenize.COMMENT for 类型, _, _, _, _ in 标记)
    assert "zhaozhiyi" not in 源码.lower()
    assert "C:\\Users" not in 源码


def test_五个Pareto点采用统一联合协调条件(monkeypatch):
    Pareto表 = pd.DataFrame(
        {
            "方案": [f"方案{i}" for i in range(5)],
            "方案类型": ["Pareto方案"] * 5,
            "EpsilonCarbon_tCO2": [np.nan] * 5,
            "EpsilonCurtailmentRate": [np.nan] * 5,
            "OperatingCost_CNY": [10.0, 20.0, 30.0, 40.0, 50.0],
            "CarbonEmission_tCO2": [50.0, 40.0, 30.0, 20.0, 10.0],
            "CurtailmentRate": [0.5, 0.4, 0.3, 0.2, 0.1],
            "RenewableUtilization": [0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    初始任务 = pd.DataFrame(
        {
            "TaskType": ["BatchInference"],
            "NetworkLatency_ms": [10.0],
            "StartHour": [1.0],
            "ArrivalHour": [0.0],
            "FinishHour": [2.0],
            "LatestFinishHour": [3.0],
            "MigrationFlag": [0],
        }
    )
    初始能源映射 = {名称: pd.DataFrame() for 名称 in Pareto表["方案"]}
    调用记录 = []

    def 模拟协调(*参数, **关键字):
        序号 = len(调用记录)
        调用记录.append((参数[0].copy(), 关键字["最大迭代"]))
        能源 = pd.DataFrame(
            {
                "Region": ["A"],
                "AvailableRenewable_MW": [10.0],
                "Curtailment_MW": [5.0 - 序号],
                "NetGridImport_MW": [float(序号)],
                "OperatingCost_CNY": [float(序号 + 1)],
                "CarbonEmission_tCO2": [float(5 - 序号)],
            }
        )
        指标 = 问题四.计算能源指标(能源)
        return 初始任务.copy(), pd.DataFrame(), 能源, 指标, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(问题四, "协调任务与能源", 模拟协调)
    协调表, _, _, _, _, _ = 问题四.联合协调Pareto方案(
        Pareto表,
        初始能源映射,
        初始任务,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert len(调用记录) == 5
    assert all(轮数 == 问题四.统一协调迭代数 for _, 轮数 in 调用记录)
    assert all(任务.equals(初始任务) for 任务, _ in 调用记录)
    assert 协调表["Selected"].sum() == 1


def test_任务约束记录使用指定情景名称():
    任务 = pd.DataFrame(
        {
            "Scheduled": [True],
            "TaskID": [1],
            "ArrivalHour": [0.0],
            "EarliestStartHour": [0.0],
            "StartHour": [0.0],
            "FinishHour": [1.0],
            "EstimatedDuration_min": [60.0],
            "TaskType": ["RealTimeInference"],
            "NetworkLatency_ms": [1.0],
            "MaxLatency_ms": [2.0],
            "LatestFinishHour": [2.0],
        }
    )
    负荷 = pd.DataFrame(
        {
            "GPU_Utilization": [0.5],
            "IT_Utilization": [0.5],
            "Facility_Utilization": [0.5],
            "Hour": [2406],
            "AI_IT_Load_MW": [0.0],
        }
    )
    结果 = 问题四.检验任务约束(任务, 负荷, "严格碳约束")
    assert set(结果["Scenario"]) == {"严格碳约束"}
    assert 结果["Passed"].all()


def test_核心结果表包含时延和服务质量字段():
    Pareto表 = pd.DataFrame(
        {
            "方案": ["方案1"],
            "方案类型": ["联合协调方案"],
            "InitialOperatingCost_CNY": [10.0],
            "OperatingCost_CNY": [8.0],
            "CarbonEmission_tCO2": [1.0],
            "CurtailmentRate": [0.1],
            "RenewableUtilization": [0.9],
            "AverageLatency_ms": [10.0],
            "P95Latency_ms": [10.0],
            "OnTimeCompletionRate": [1.0],
            "MaxRegionalPeak_MW": [0.0],
            "CoordinatedNonDominated": [True],
            "IdealPointDistance": [0.0],
            "Selected": [True],
        }
    )
    任务 = pd.DataFrame(
        {
            "TaskID": [1],
            "TaskType": ["RealTimeInference"],
            "SourceRegion": ["A"],
            "ExecutionRegion": ["A"],
            "NetworkLatency_ms": [10.0],
            "StartHour": [0.0],
            "ArrivalHour": [0.0],
            "FinishHour": [1.0],
            "LatestFinishHour": [2.0],
            "MigrationFlag": [0],
        }
    )
    能源 = pd.DataFrame(
        {
            "Region": ["A"],
            "Hour": [0],
            "AvailableRenewable_MW": [10.0],
            "Curtailment_MW": [1.0],
            "NetGridImport_MW": [0.0],
            "OperatingCost_CNY": [8.0],
            "CarbonEmission_tCO2": [1.0],
            "ChargePower_MW": [0.0],
            "DischargePower_MW": [0.0],
        }
    )
    Pareto输出, 基准表, 情景表 = 问题四.构造核心结果表(Pareto表, 任务, 能源, [])
    assert {"平均网络时延_ms", "按时完成率", "最大区域峰值_MW"}.issubset(Pareto输出.columns)
    assert {
        "平均网络时延_ms",
        "P95网络时延_ms",
        "最大网络时延_ms",
        "迁移率_%",
        "实时任务立即开工率_%",
        "按时完成率_%",
        "平均等待时间_h",
        "P95等待时间_h",
    }.issubset(情景表.columns)
    assert "最大网络时延" in set(基准表["指标"])
