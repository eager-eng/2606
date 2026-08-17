import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from 问题二 import 执行问题二调度, 构造指标对比表, 检验问题二约束, 计算三目标综合值, 计算能源状态, 计算边际能源状态, 选择最优候选


def test_边际能源状态在零负荷变化时复现附件基准():
    结果 = 计算边际能源状态(
        设施负荷=np.array([100.0]),
        基准设施负荷=np.array([100.0]),
        可用新能源=np.array([80.0]),
        基准购电=np.array([50.0]),
        基准售电=np.array([10.0]),
        基准弃电=np.array([20.0]),
        基准充电=np.array([0.0]),
        基准放电=np.array([0.0]),
        购电价=np.array([500.0]),
        售电价=np.array([100.0]),
        碳强度=np.array([0.5]),
    )
    np.testing.assert_allclose(结果["GridPurchase_MW"], [50.0])
    np.testing.assert_allclose(结果["GridSell_MW"], [10.0])
    np.testing.assert_allclose(结果["Curtailment_MW"], [20.0])
    np.testing.assert_allclose(结果["OperatingCost_CNY"], [24000.0])
    np.testing.assert_allclose(结果["CarbonEmission_tCO2"], [25.0])


def test_边际能源状态只按负荷增量调整弃电售电和购电():
    公共参数 = {
        "基准设施负荷": np.array([100.0, 100.0, 100.0]),
        "可用新能源": np.array([80.0, 80.0, 80.0]),
        "基准购电": np.array([50.0, 50.0, 50.0]),
        "基准售电": np.array([10.0, 10.0, 10.0]),
        "基准弃电": np.array([20.0, 20.0, 20.0]),
        "基准充电": np.zeros(3),
        "基准放电": np.zeros(3),
        "购电价": np.full(3, 500.0),
        "售电价": np.full(3, 100.0),
        "碳强度": np.full(3, 0.5),
    }
    结果 = 计算边际能源状态(设施负荷=np.array([125.0, 140.0, 40.0]), **公共参数)
    np.testing.assert_allclose(结果["GridPurchase_MW"], [50.0, 60.0, 0.0])
    np.testing.assert_allclose(结果["GridSell_MW"], [5.0, 0.0, 10.0])
    np.testing.assert_allclose(结果["Curtailment_MW"], [0.0, 0.0, 30.0])
    np.testing.assert_allclose(结果["PowerBalanceResidual_MW"], [0.0, 0.0, 0.0], atol=1e-12)


def test_能源状态满足购售电和弃电口径():
    结果 = 计算能源状态(
        设施负荷=np.array([100.0, 40.0]),
        可用新能源=np.array([80.0, 100.0]),
        基准充电=np.array([10.0, 0.0]),
        基准放电=np.array([20.0, 0.0]),
        基准售电=np.array([0.0, 20.0]),
        购电价=np.array([500.0, 500.0]),
        售电价=np.array([100.0, 100.0]),
        碳强度=np.array([0.5, 0.5]),
    )
    np.testing.assert_allclose(结果["GridPurchase_MW"], [10.0, 0.0])
    np.testing.assert_allclose(结果["GridSell_MW"], [0.0, 20.0])
    np.testing.assert_allclose(结果["Curtailment_MW"], [0.0, 40.0])
    np.testing.assert_allclose(结果["OperatingCost_CNY"], [5000.0, -2000.0])
    np.testing.assert_allclose(结果["CarbonEmission_tCO2"], [5.0, 0.0])
    np.testing.assert_allclose(结果["PowerBalanceResidual_MW"], [0.0, 0.0], atol=1e-12)
    assert abs(结果["RenewableUtilization"] - 140.0 / 180.0) < 1e-12


def test_三目标综合值使用基准相对变化且新能源为正向指标():
    得分 = 计算三目标综合值(
        运行成本=900.0,
        碳排放=90.0,
        新能源利用率=0.55,
        基准成本=1000.0,
        基准碳排放=100.0,
        基准新能源利用率=0.5,
    )
    assert abs(得分 + 0.1) < 1e-12


def test_指标对比表中的新能源利用率按百分数输出():
    指标 = {
        "基准运行成本_CNY": 1000.0,
        "优化运行成本_CNY": 900.0,
        "基准碳排放_tCO2": 100.0,
        "优化碳排放_tCO2": 90.0,
        "基准新能源利用率": 0.5,
        "优化新能源利用率": 0.55,
        "基准平均网络时延_ms": 5.0,
        "优化平均网络时延_ms": 10.0,
        "基准P95网络时延_ms": 5.0,
        "优化P95网络时延_ms": 15.0,
        "基准最大网络时延_ms": 5.0,
        "优化最大网络时延_ms": 20.0,
    }
    对比 = 构造指标对比表(指标).set_index("指标")
    assert np.isclose(对比.loc["新能源利用率", "基准值"], 50.0)
    assert np.isclose(对比.loc["新能源利用率", "优化值"], 55.0)
    assert 对比.loc["新能源利用率", "单位"] == "%"


def test_候选选择先比较三目标再依次比较时延开工和迁移():
    候选 = [
        {"Score": -0.2, "NetworkLatency_ms": 20.0, "StartHour": 4, "MigrationFlag": 1, "Region": "B"},
        {"Score": -0.2, "NetworkLatency_ms": 10.0, "StartHour": 5, "MigrationFlag": 1, "Region": "C"},
        {"Score": -0.2, "NetworkLatency_ms": 10.0, "StartHour": 3, "MigrationFlag": 0, "Region": "A"},
        {"Score": -0.1, "NetworkLatency_ms": 5.0, "StartHour": 1, "MigrationFlag": 0, "Region": "D"},
    ]
    最优 = 选择最优候选(候选)
    assert 最优["Region"] == "A"


def 构造双区域样例():
    任务 = pd.DataFrame(
        [
            [1, "AITraining", 0, 10, 60, "Low", "A", 100, 3, 0],
            [2, "RealTimeInference", 1, 5, 60, "High", "A", 5, 2, 1],
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
        ],
    )
    GPU信息 = pd.DataFrame(
        {
            "Region": ["A", "B"],
            "Available_GPU": [100, 100],
            "Max_IT_Power_MW": [100.0, 100.0],
            "PUE": [1.0, 1.0],
            "Max_Facility_Power_MW": [100.0, 100.0],
        }
    )
    网络 = pd.DataFrame(
        {
            "FromRegion": ["A", "A", "B", "B"],
            "ToRegion": ["A", "B", "A", "B"],
            "NetworkLatency_ms": [5.0, 20.0, 20.0, 5.0],
        }
    )
    功率映射 = pd.DataFrame(
        {
            "TaskType": ["AITraining", "RealTimeInference"],
            "GPU_Power_MW_per_EquivalentGPU": [1.0, 1.0],
        }
    )
    行 = []
    for 小时 in range(5):
        for 区域 in ["A", "B"]:
            行.append(
                {
                    "Hour": 小时,
                    "Region": 区域,
                    "ElectricityPrice_CNY_per_MWh": 1000.0,
                    "SellPrice_CNY_per_MWh": 0.0,
                    "CarbonIntensity_tCO2_per_MWh": 1.0,
                    "AvailableRenewable_MW": 100.0 if 区域 == "B" else 0.0,
                    "UsedRenewable_MW": 10.0 if 区域 == "B" else 0.0,
                    "RenewableCharge_MW": 0.0,
                    "NonAI_IT_Load_MW": 10.0 if 区域 == "B" else 1.0,
                    "Baseline_AI_IT_Load_MW": 0.0,
                    "Total_Load_MW": 10.0 if 区域 == "B" else 1.0,
                    "ChargePower_MW": 0.0,
                    "DischargePower_MW": 0.0,
                    "GridSell_MW": 0.0,
                    "GridPurchase_MW": 0.0 if 区域 == "B" else 1.0,
                    "Curtailment_MW": 90.0 if 区域 == "B" else 0.0,
                    "CarbonEmission_tCO2": 0.0 if 区域 == "B" else 1.0,
                }
            )
    return 任务, GPU信息, pd.DataFrame(行), 网络, 功率映射


def test_问题二调度以三目标选址且实时任务受时延和即时开工约束():
    任务, GPU信息, 时间数据, 网络, 功率映射 = 构造双区域样例()
    调度, 资源, 指标 = 执行问题二调度(任务, GPU信息, 时间数据, 网络, 功率映射, 截止小时=4, 候选数=2)
    结果 = 调度.set_index("TaskID")
    assert 结果.loc[1, "ExecutionRegion"] == "B"
    assert 结果.loc[1, "StartHour"] == 0
    assert 结果.loc[2, "ExecutionRegion"] == "A"
    assert 结果.loc[2, "StartHour"] == 1
    assert 调度["Scheduled"].all()
    assert 指标["任务数"] == 2
    assert 指标["基准平均网络时延_ms"] == 5.0
    assert 指标["优化平均网络时延_ms"] == 12.5
    assert len(资源) == 10


def test_问题二约束检验覆盖调度容量和电力平衡():
    任务, GPU信息, 时间数据, 网络, 功率映射 = 构造双区域样例()
    调度, 资源, _ = 执行问题二调度(任务, GPU信息, 时间数据, 网络, 功率映射, 截止小时=4, 候选数=2)
    检验 = 检验问题二约束(任务, 调度, 资源, 截止小时=4)
    assert 检验["是否通过"].all()
    assert 检验["违约数或残差"].max() == 0


def test_问题二约束检验识别提前开工和执行时长错误():
    任务, GPU信息, 时间数据, 网络, 功率映射 = 构造双区域样例()
    调度, 资源, _ = 执行问题二调度(任务, GPU信息, 时间数据, 网络, 功率映射, 截止小时=4, 候选数=2)
    错误调度 = 调度.copy()
    错误调度.loc[错误调度["TaskID"] == 1, "StartHour"] = -1.0
    检验 = 检验问题二约束(任务, 错误调度, 资源, 截止小时=4).set_index("检验项")
    assert 检验.loc["任务早于允许时刻开工", "违约数或残差"] == 1
    assert 检验.loc["任务连续执行时长不一致", "违约数或残差"] == 1


def test_调度负荷等于基准负荷时指标必须恢复附件结果():
    任务 = pd.DataFrame(
        [[1, "RealTimeInference", 0, 10, 60, "High", "A", 10, 1, 0]],
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
        ],
    )
    GPU信息 = pd.DataFrame(
        {
            "Region": ["A"],
            "Available_GPU": [100],
            "Max_IT_Power_MW": [100.0],
            "PUE": [1.0],
            "Max_Facility_Power_MW": [100.0],
        }
    )
    网络 = pd.DataFrame({"FromRegion": ["A"], "ToRegion": ["A"], "NetworkLatency_ms": [5.0]})
    功率映射 = pd.DataFrame(
        {"TaskType": ["RealTimeInference"], "GPU_Power_MW_per_EquivalentGPU": [1.0]}
    )
    时间数据 = pd.DataFrame(
        [
            {
                "Hour": 0,
                "Region": "A",
                "ElectricityPrice_CNY_per_MWh": 100.0,
                "SellPrice_CNY_per_MWh": 0.0,
                "CarbonIntensity_tCO2_per_MWh": 1.0,
                "AvailableRenewable_MW": 10.0,
                "UsedRenewable_MW": 5.0,
                "RenewableCharge_MW": 0.0,
                "NonAI_IT_Load_MW": 0.0,
                "Baseline_AI_IT_Load_MW": 10.0,
                "Total_Load_MW": 10.0,
                "ChargePower_MW": 0.0,
                "DischargePower_MW": 0.0,
                "GridSell_MW": 0.0,
                "GridPurchase_MW": 5.0,
                "Curtailment_MW": 5.0,
                "CarbonEmission_tCO2": 5.0,
            },
            {
                "Hour": 1,
                "Region": "A",
                "ElectricityPrice_CNY_per_MWh": 100.0,
                "SellPrice_CNY_per_MWh": 0.0,
                "CarbonIntensity_tCO2_per_MWh": 1.0,
                "AvailableRenewable_MW": 0.0,
                "UsedRenewable_MW": 0.0,
                "RenewableCharge_MW": 0.0,
                "NonAI_IT_Load_MW": 0.0,
                "Baseline_AI_IT_Load_MW": 0.0,
                "Total_Load_MW": 0.0,
                "ChargePower_MW": 0.0,
                "DischargePower_MW": 0.0,
                "GridSell_MW": 0.0,
                "GridPurchase_MW": 0.0,
                "Curtailment_MW": 0.0,
                "CarbonEmission_tCO2": 0.0,
            },
        ]
    )
    _, 资源, 指标 = 执行问题二调度(任务, GPU信息, 时间数据, 网络, 功率映射, 截止小时=1, 候选数=1)
    assert 指标["优化运行成本_CNY"] == 指标["基准运行成本_CNY"] == 500.0
    assert 指标["优化碳排放_tCO2"] == 指标["基准碳排放_tCO2"] == 5.0
    assert 指标["优化新能源利用率"] == 指标["基准新能源利用率"] == 0.5
    np.testing.assert_allclose(资源["GridPurchase_MW"], [5.0, 0.0])
    np.testing.assert_allclose(资源["Curtailment_MW"], [5.0, 0.0])
