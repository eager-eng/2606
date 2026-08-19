import json
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


根目录 = Path(__file__).resolve().parents[1]
try:
    import gurobipy
except ModuleNotFoundError:
    虚拟Gurobi = types.ModuleType("gurobipy")
    虚拟Gurobi.GRB = types.SimpleNamespace()
    虚拟Gurobi.Model = object
    sys.modules["gurobipy"] = 虚拟Gurobi
模块规格 = importlib.util.spec_from_file_location("问题四", 根目录 / "code" / "问题四.py")
问题四 = importlib.util.module_from_spec(模块规格)
模块规格.loader.exec_module(问题四)
输出目录 = 根目录 / "outputs" / "问题四计算结果"
图片目录 = 根目录 / "figures" / "问题四计算结果"
报告路径 = 根目录 / "reports" / "问题四计算结果.md"
Pareto表 = pd.read_csv(输出目录 / "Pareto方案及折中方案.csv")
基准综合表 = pd.read_csv(输出目录 / "基准折中方案综合结果.csv")
情景表 = pd.read_csv(输出目录 / "单因素情景比较.csv")
检验结果 = pd.read_csv(输出目录 / "约束检验.csv")
区域逐时 = pd.read_csv(输出目录 / "区域逐时协同运行.csv")
基准能源 = 区域逐时[区域逐时["Scenario"] == "基准折中"].copy()
基准任务 = pd.read_csv(输出目录 / "最终任务调度方案.csv")
迁移明细 = pd.read_csv(输出目录 / "任务调整明细.csv")
情景任务映射 = 问题四.恢复情景任务映射(基准任务, 情景表, 迁移明细)
问题四.绘制问题四图表(Pareto表, 基准能源, 情景表, 情景任务映射, 图片目录)
运行统计路径 = 输出目录 / "运行统计.json"
运行统计 = json.loads(运行统计路径.read_text(encoding="utf-8"))
运行统计.pop("Pareto非支配方案数", None)
运行统计["Pareto联合协调方案数"] = int(len(Pareto表))
运行统计["约束检验通过数"] = int(检验结果["Passed"].astype(bool).sum())
运行统计["约束检验总数"] = int(len(检验结果))
运行统计路径.write_text(json.dumps(运行统计, ensure_ascii=False, indent=2), encoding="utf-8")
问题四.生成问题四报告(
    Pareto表,
    基准综合表,
    情景表,
    检验结果,
    float(运行统计["总运行时间_s"]),
    报告路径,
)
print(json.dumps(运行统计, ensure_ascii=False, indent=2))
