from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from 问题一 import (
    SBA克罗斯顿预测,
    ARIMA预测,
    执行基础调度,
    检验调度约束,
    生成结果报告,
    构造日内GPU总需求,
    构造到达需求序列,
    提取最后24小时结果,
    计算预测指标,
    计算小时重叠,
    选择预测模型,
)


def test_计算跨小时任务的实际重叠时长():
    assert 计算小时重叠(10, 11.5, 10) == 1.0
    assert 计算小时重叠(10, 11.5, 11) == 0.5
    assert 计算小时重叠(10, 11.5, 12) == 0.0


def test_到达需求按小时区域类型聚合并补零():
    任务 = pd.DataFrame(
        {
            "ArrivalHour": [0, 0, 1],
            "SourceRegion": ["A", "A", "B"],
            "TaskType": ["X", "X", "Y"],
            "GPU_Demand": [2, 3, 4],
        }
    )
    序列 = 构造到达需求序列(任务, 3, ["A", "B"], ["X", "Y"])
    assert 序列.loc[0, ("A", "X")] == 5
    assert 序列.loc[1, ("B", "Y")] == 4
    assert 序列.loc[2, ("A", "Y")] == 0


def test_日内统计先汇总任务类型再对日期求平均():
    列 = pd.MultiIndex.from_product(
        [["A", "B"], ["X", "Y"]], names=["SourceRegion", "TaskType"]
    )
    序列 = pd.DataFrame(0.0, index=range(48), columns=列)
    序列.loc[0, ("A", "X")] = 2
    序列.loc[0, ("A", "Y")] = 3
    序列.loc[24, ("A", "X")] = 4
    序列.loc[24, ("A", "Y")] = 5
    结果 = 构造日内GPU总需求(序列)
    assert 结果.loc["A", 0] == 7
    assert 结果.loc["A", 1] == 0
    assert 结果.loc["B", 0] == 0


def test_最后24小时图严格排除收尾时域():
    调度 = pd.DataFrame(
        {"ArrivalHour": [2375, 2376, 2399, 2400], "Scheduled": [True, True, True, True]}
    )
    资源 = pd.DataFrame({"Hour": [2375, 2376, 2399, 2400], "Region": ["A", "A", "A", "A"]})
    末调度, 末资源 = 提取最后24小时结果(调度, 资源)
    assert 末调度["ArrivalHour"].tolist() == [2376, 2399]
    assert 末资源["Hour"].tolist() == [2376, 2399]


def test_按训练段零值率执行双轨分流():
    assert 选择预测模型(pd.Series([0, 0, 0, 1])) == "SBA-Croston"
    assert 选择预测模型(pd.Series([1, 0, 1, 1])) == "ARIMA"


def test_SBA克罗斯顿输出长度正确且非负():
    预测 = SBA克罗斯顿预测(pd.Series([0, 0, 4, 0, 0, 6, 0]), 4, 0.1)
    assert len(预测) == 4
    assert (预测 >= 0).all()
    assert 预测.nunique() == 1


def test_预测指标计算口径():
    指标 = 计算预测指标(pd.Series([0, 2]), pd.Series([1, 1]))
    assert 指标["MAE"] == 1
    assert 指标["RMSE"] == 1
    assert 指标["WMAPE"] == 1


def test_ARIMA预测长度正确且非负():
    序列 = pd.Series([10 + i * 0.2 + (-1) ** i for i in range(80)])
    结果 = ARIMA预测(序列, 6, 最大阶数=1)
    assert len(结果["预测值"]) == 6
    assert (结果["预测值"] >= 0).all()
    assert len(结果["阶数"]) == 3


def test_调度优先保证实时任务并选择均衡区域():
    任务 = pd.DataFrame(
        [
            ["F", "BatchInference", 0, 8, 60, "Medium", "A", 10, 3, 0, "NonPreemptive"],
            ["R", "RealTimeInference", 0, 5, 60, "High", "A", 10, 1, 0, "NonPreemptive"],
        ],
        columns=[
            "TaskID",
            "TaskType",
            "ArrivalHour",
            "GPU_Demand",
            "EstimatedDuration_min",
            "DelaySensitivity",
            "SourceRegion",
            "MaxLatency_ms",
            "LatestFinishHour",
            "EarliestStartHour",
            "ExecutionMode",
        ],
    )
    GPU信息 = pd.DataFrame(
        {
            "Region": ["A", "B"],
            "Available_GPU": [10, 10],
            "Max_IT_Power_MW": [10, 10],
            "PUE": [1, 1],
            "Max_Facility_Power_MW": [10, 10],
        }
    )
    时间数据 = pd.DataFrame(
        [(小时, 区域, 0) for 小时 in range(3) for 区域 in ["A", "B"]],
        columns=["Hour", "Region", "NonAI_IT_Load_MW"],
    )
    网络 = pd.DataFrame(
        [(来源, 目标, 0) for 来源 in ["A", "B"] for 目标 in ["A", "B"]],
        columns=["FromRegion", "ToRegion", "NetworkLatency_ms"],
    )
    功率 = pd.DataFrame(
        {"TaskType": ["RealTimeInference", "BatchInference"], "GPU_Power_MW_per_EquivalentGPU": [1, 1]}
    )
    调度, _ = 执行基础调度(任务, GPU信息, 时间数据, 网络, 功率, 3)
    调度 = 调度.set_index("TaskID")
    assert 调度.loc["R", "StartHour"] == 0
    assert 调度.loc["R", "ExecutionRegion"] == "A"
    assert 调度.loc["F", "StartHour"] == 0
    assert 调度.loc["F", "ExecutionRegion"] == "B"


def test_约束检验只汇总必要硬约束():
    任务 = pd.DataFrame(
        {
            "TaskID": ["R"],
            "TaskType": ["RealTimeInference"],
            "ArrivalHour": [0],
            "LatestFinishHour": [1],
            "MaxLatency_ms": [10],
        }
    )
    调度 = pd.DataFrame(
        {
            "TaskID": ["R"],
            "ExecutionRegion": ["A"],
            "StartHour": [0.0],
            "FinishHour": [1.0],
            "NetworkLatency_ms": [0.0],
            "Scheduled": [True],
        }
    )
    资源 = pd.DataFrame(
        {"GPU_Utilization": [1.1], "IT_Utilization": [0.8], "Facility_Utilization": [0.7]}
    )
    检验 = 检验调度约束(任务, 调度, 资源, 1).set_index("检验项")
    assert 检验.loc["GPU容量超限", "违约数"] == 1
    assert 检验.loc["实时任务未立即开工", "违约数"] == 0
    assert 检验.loc["任务遗漏或未排程", "违约数"] == 0


def test_结果报告不依赖额外表格包(tmp_path):
    路径 = tmp_path / "结果.md"
    任务 = pd.DataFrame({"SourceRegion": ["A"], "TaskType": ["X"]})
    指标 = pd.DataFrame({"Model": ["ARIMA"], "MAE": [1.0], "RMSE": [1.0], "WMAPE": [1.0]})
    调度 = pd.DataFrame({"Scheduled": [True]})
    资源 = pd.DataFrame({"Region": ["A"], "GPU_Utilization": [0.5]})
    检验 = pd.DataFrame({"检验项": ["GPU容量超限"], "违约数": [0], "是否通过": [True]})
    生成结果报告(路径, 任务, 指标, 调度, 资源, 检验, 1.0)
    assert "| GPU容量超限 | 0 | 是 |" in 路径.read_text(encoding="utf-8")
