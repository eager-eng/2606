import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from 问题三 import 构造无储能方案, 求解储能窗口, 计算区域峰值波动


def test_无储能方案优先使用新能源并受售电上限约束():
    时段数据 = pd.DataFrame(
        {
            "Hour": [0, 1],
            "Region": ["RegionA", "RegionA"],
            "Facility_Load_MW": [10.0, 10.0],
            "AvailableRenewable_MW": [5.0, 20.0],
            "ElectricityPrice_CNY_per_MWh": [100.0, 100.0],
            "SellPrice_CNY_per_MWh": [50.0, 50.0],
            "CarbonIntensity_tCO2_per_MWh": [0.5, 0.5],
        }
    )
    储能参数 = {"SellLimit_MW": 4.0, "MaxGridExport_MW": 4.0, "InitialSOC_MWh": 2.0}
    结果 = 构造无储能方案(时段数据, 储能参数)
    np.testing.assert_allclose(结果["GridPurchase_MW"], [5.0, 0.0])
    np.testing.assert_allclose(结果["GridSell_MW"], [0.0, 4.0])
    np.testing.assert_allclose(结果["Curtailment_MW"], [0.0, 6.0])
    np.testing.assert_allclose(结果["SOC_MWh"], [2.0, 2.0])
    np.testing.assert_allclose(结果["PowerBalanceResidual_MW"], [0.0, 0.0], atol=1e-9)


def test_最低成本窗口利用低价充电并遵守峰值安全约束():
    时段数据 = pd.DataFrame(
        {
            "Hour": [0, 1],
            "Region": ["RegionA", "RegionA"],
            "Facility_Load_MW": [10.0, 10.0],
            "AvailableRenewable_MW": [0.0, 0.0],
            "ElectricityPrice_CNY_per_MWh": [1.0, 10.0],
            "SellPrice_CNY_per_MWh": [0.0, 0.0],
            "CarbonIntensity_tCO2_per_MWh": [1.0, 1.0],
        }
    )
    储能参数 = {
        "StorageCapacity_MWh": 10.0,
        "MinSOC_MWh": 0.0,
        "InitialSOC_MWh": 0.0,
        "MaxChargePower_MW": 10.0,
        "MaxDischargePower_MW": 10.0,
        "ChargeEfficiency": 1.0,
        "DischargeEfficiency": 1.0,
        "SellLimit_MW": 0.0,
        "MaxGridImport_MW": 20.0,
        "MaxGridExport_MW": 0.0,
    }
    结果, 统计 = 求解储能窗口(
        时段数据,
        储能参数,
        场景="最低成本方案",
        初始SOC_MWh=0.0,
        终端SOC下限_MWh=0.0,
        峰值上限_MW=15.0,
        基准成本_CNY=110.0,
        基准碳排放_tCO2=20.0,
    )
    np.testing.assert_allclose(结果["GridPurchase_MW"], [15.0, 5.0], atol=1e-6)
    np.testing.assert_allclose(结果["ChargePower_MW"], [5.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(结果["DischargePower_MW"], [0.0, 5.0], atol=1e-6)
    assert 结果["NetGridImport_MW"].max() <= 15.0 + 1e-6
    assert abs(结果["SOC_MWh"].iloc[-1]) <= 1e-6
    assert 统计["求解状态"] == "OPTIMAL"


def test_区域峰值波动指标按净购电曲线计算():
    逐时结果 = pd.DataFrame(
        {
            "Scenario": ["测试方案"] * 3,
            "Region": ["RegionA"] * 3,
            "Hour": [0, 1, 2],
            "NetGridImport_MW": [10.0, 20.0, 30.0],
        }
    )
    基准指标 = pd.DataFrame(
        {
            "Region": ["RegionA"],
            "PeakNetImport_MW": [40.0],
            "NetImportStd_MW": [10.0],
            "MeanHourlyRamp_MW_per_h": [15.0],
            "PeakValleyRange_MW": [30.0],
        }
    )
    指标 = 计算区域峰值波动(逐时结果, 基准指标).iloc[0]
    assert 指标["PeakNetImport_MW"] == 30.0
    assert 指标["PeakHour_h"] == 2
    assert abs(指标["PeakReduction_MW"] - 10.0) < 1e-12
    assert abs(指标["PeakReductionRate"] - 0.25) < 1e-12
    assert abs(指标["NetImportStd_MW"] - np.sqrt(200.0 / 3.0)) < 1e-12
    assert abs(指标["FluctuationReductionRate"] - (1.0 - np.sqrt(200.0 / 3.0) / 10.0)) < 1e-12
    assert 指标["MeanHourlyRamp_MW_per_h"] == 10.0
    assert abs(指标["MeanHourlyRampReductionRate"] - 1.0 / 3.0) < 1e-12
    assert 指标["PeakValleyRange_MW"] == 20.0
    assert abs(指标["PeakValleyReductionRate"] - 1.0 / 3.0) < 1e-12


def test_储能窗口结果满足能量平衡和充放电互斥():
    时段数据 = pd.DataFrame(
        {
            "Hour": [0, 1, 2],
            "Region": ["RegionE"] * 3,
            "Facility_Load_MW": [12.0, 12.0, 12.0],
            "AvailableRenewable_MW": [20.0, 0.0, 5.0],
            "ElectricityPrice_CNY_per_MWh": [2.0, 8.0, 4.0],
            "SellPrice_CNY_per_MWh": [1.0, 1.0, 1.0],
            "CarbonIntensity_tCO2_per_MWh": [0.2, 0.8, 0.4],
        }
    )
    储能参数 = {
        "StorageCapacity_MWh": 8.0,
        "MinSOC_MWh": 0.0,
        "InitialSOC_MWh": 2.0,
        "MaxChargePower_MW": 4.0,
        "MaxDischargePower_MW": 4.0,
        "ChargeEfficiency": 0.9,
        "DischargeEfficiency": 0.9,
        "SellLimit_MW": 3.0,
        "MaxGridImport_MW": 20.0,
        "MaxGridExport_MW": 3.0,
    }
    结果, _ = 求解储能窗口(
        时段数据,
        储能参数,
        场景="成本—碳排放等权方案",
        初始SOC_MWh=2.0,
        终端SOC下限_MWh=2.0,
        峰值上限_MW=20.0,
        基准成本_CNY=100.0,
        基准碳排放_tCO2=10.0,
    )
    assert 结果["PowerBalanceResidual_MW"].abs().max() <= 1e-6
    assert ((结果["ChargePower_MW"] > 1e-7) & (结果["DischargePower_MW"] > 1e-7)).sum() == 0
    assert 结果["SOC_MWh"].between(0.0 - 1e-7, 8.0 + 1e-7).all()
    assert 结果["SOC_MWh"].iloc[-1] >= 2.0 - 1e-7


def test_区域融合表直接给出相对无储能的储能增量影响():
    逐时结果 = pd.DataFrame(
        {
            "Scenario": ["无储能辅助对照"] * 3 + ["成本—碳排放等权方案"] * 3,
            "Region": ["RegionD"] * 6,
            "Hour": [0, 1, 2, 0, 1, 2],
            "NetGridImport_MW": [0.0, 10.0, 20.0, 5.0, 10.0, 15.0],
        }
    )
    基准指标 = pd.DataFrame(
        {
            "Region": ["RegionD"],
            "PeakNetImport_MW": [20.0],
            "NetImportStd_MW": [np.sqrt(200.0 / 3.0)],
            "MeanHourlyRamp_MW_per_h": [10.0],
            "PeakValleyRange_MW": [20.0],
        }
    )
    指标 = 计算区域峰值波动(逐时结果, 基准指标)
    等权 = 指标[指标["Scenario"] == "成本—碳排放等权方案"].iloc[0]
    assert 等权["StoragePeakReduction_MW"] == 5.0
    assert abs(等权["StorageNetImportStdReduction_MW"] - np.sqrt(200.0 / 3.0) / 2.0) < 1e-12
    assert 等权["StorageMeanHourlyRampReduction_MW_per_h"] == 5.0
    assert 等权["StoragePeakValleyReduction_MW"] == 10.0


def test_全时段净外送时峰值净购电按零计算():
    逐时结果 = pd.DataFrame(
        {
            "Scenario": ["无储能辅助对照"] * 3 + ["成本—碳排放等权方案"] * 3,
            "Region": ["RegionD"] * 6,
            "Hour": [0, 1, 2, 0, 1, 2],
            "NetGridImport_MW": [-100.0, -50.0, -80.0, -180.0, -180.0, -180.0],
        }
    )
    基准指标 = pd.DataFrame(
        {
            "Region": ["RegionD"],
            "PeakNetImport_MW": [200.0],
            "NetImportStd_MW": [10.0],
            "MeanHourlyRamp_MW_per_h": [10.0],
            "PeakValleyRange_MW": [20.0],
        }
    )
    指标 = 计算区域峰值波动(逐时结果, 基准指标)
    assert (指标["PeakNetImport_MW"] == 0.0).all()
    等权 = 指标[指标["Scenario"] == "成本—碳排放等权方案"].iloc[0]
    assert 等权["StoragePeakReduction_MW"] == 0.0
