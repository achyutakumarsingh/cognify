import sys
import yaml
import json
import pandas as pd
from pathlib import Path

# Setup Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.helpers import setup_logger
from src.models.data_loader import ForecastDataLoader

from src.simulation.cost_model import CostModel
from src.simulation.inventory_simulator import InventorySimulator
from src.simulation.scenario_engine import ScenarioEngine
from src.simulation.business_impact_analyzer import BusinessImpactAnalyzer
from src.simulation.business_visualizer import BusinessVisualizer

logger = setup_logger("supply_chain")

def main():
    logger.info("====================================================================")
    logger.info("STAGE 7 │ Inventory Simulation & Business Impact Engine")
    logger.info("====================================================================")
    
    # Load config
    config_path = Path("config/simulation.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    logger.info("[Step 1/7] Loading Stage 6 Dashboard Data and Features...")
    df_dashboard = pd.read_parquet(config["inputs"]["dashboard_data"])
    
    loader = ForecastDataLoader()
    # Need test features for price and scenario flags
    X_test, _ = loader.load_split("test", dev_mode=True)
    
    if len(X_test) != len(df_dashboard):
        logger.warning(f"Length mismatch: X_test={len(X_test)}, Dash={len(df_dashboard)}. Truncating...")
        X_test = X_test.iloc[:len(df_dashboard)].reset_index(drop=True)
        
    logger.info("[Step 2/7] Running Inventory Simulator (Physical Units)...")
    simulator = InventorySimulator(config)
    df_sim = simulator.apply_policies(df_dashboard, X_test)
    
    logger.info("[Step 3/7] Applying Financial Cost Model...")
    cost_model = CostModel(config)
    df_sim = cost_model.evaluate_financials(df_sim, "Baseline")
    df_sim = cost_model.evaluate_financials(df_sim, "Proposed")
    
    logger.info("[Step 4/7] Computing Global KPIs and ROI...")
    analyzer = BusinessImpactAnalyzer()
    global_report = analyzer.generate_comparison_report(df_sim)
    
    logger.info("[Step 5/7] Evaluating Demand Scenarios & Sensitivity Analysis...")
    scenario_engine = ScenarioEngine(config)
    scenario_results = scenario_engine.evaluate_scenarios(df_sim)
    sensitivity_results = scenario_engine.run_sensitivity_sweep(df_sim)
    
    logger.info("[Step 6/7] Generating Business Visualizations...")
    visualizer = BusinessVisualizer(config["outputs"]["plots_dir"])
    visualizer.plot_cost_breakdown(global_report, config["plots"]["cost_breakdown"])
    visualizer.plot_service_level_improvement(global_report, config["plots"]["service_level_vs_holding"])
    visualizer.plot_scenario_performance(scenario_results, config["plots"]["scenario_performance"])
    visualizer.plot_sensitivity_analysis(sensitivity_results, config["plots"]["sensitivity_analysis"])
    
    logger.info("[Step 7/7] Saving Stage 7 Artifacts...")
    sim_dir = Path(config["outputs"]["simulation_dir"])
    sim_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(config["outputs"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the dashboard dataset (combines Stage 6 + Stage 7 financial columns)
    df_sim.to_parquet(sim_dir / config["artifacts"]["business_dataset"], index=False)
    
    # Save JSON reports
    with open(rep_dir / config["artifacts"]["financial_summary"], "w") as f:
        json.dump(global_report, f, indent=4)
        
    with open(rep_dir / config["artifacts"]["sensitivity_summary"], "w") as f:
        json.dump({"Sensitivity": sensitivity_results, "Scenarios": scenario_results}, f, indent=4)
        
    logger.info("  ✓ " + config["plots"]["cost_breakdown"])
    logger.info("  ✓ " + config["plots"]["service_level_vs_holding"])
    logger.info("  ✓ " + config["plots"]["scenario_performance"])
    logger.info("  ✓ " + config["plots"]["sensitivity_analysis"])
    logger.info("  ✓ " + config["artifacts"]["business_dataset"])
    logger.info("  ✓ " + config["artifacts"]["financial_summary"])
    
    logger.info("--------------------------------------------------------------------")
    logger.info("EXECUTIVE SUMMARY (GLOBAL IMPACT):")
    logger.info(f"  Total Cost Savings   : ${global_report['Impact']['Net_Savings']:,.2f}")
    logger.info(f"  Cost Reduction %     : {global_report['Impact']['Cost_Reduction_Pct']*100:.1f}%")
    logger.info(f"  Service Level Change : {global_report['Baseline']['Service_Level']*100:.1f}% -> {global_report['Proposed']['Service_Level']*100:.1f}%")
    logger.info(f"  Stockout Reduction   : {global_report['Impact']['Stockout_Reduction_Units']:,.0f} units")
    logger.info(f"  Implied ROI          : {global_report['Impact']['ROI']:.2f}x")
    logger.info("====================================================================")
    logger.info("STAGE 7 COMPLETE ✓")
    logger.info("====================================================================")

if __name__ == "__main__":
    main()
