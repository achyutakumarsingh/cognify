import sys
import yaml
import json
import pandas as pd
from pathlib import Path

# Setup Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.helpers import setup_logger
from src.models.data_loader import ForecastDataLoader

from src.decision.risk_scorer import RiskScorer
from src.decision.risk_classifier import RiskClassifier
from src.decision.recommendation_engine import RecommendationEngine
from src.decision.decision_visualizer import DecisionVisualizer

logger = setup_logger("supply_chain")

def main():
    logger.info("====================================================================")
    logger.info("STAGE 6 │ Supply Chain Risk Triage & Decision Intelligence Engine")
    logger.info("====================================================================")
    
    # Load config
    config_path = Path("config/decision.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    logger.info("[Step 1/6] Loading Stage 4/5 Artifacts and Test Features...")
    df_preds = pd.read_parquet(config["inputs"]["predictions"])
    df_segments = pd.read_csv(config["inputs"]["segment_analysis"])
    
    loader = ForecastDataLoader()
    # Loading dev_mode to match the size of Stage 4 and 5 data
    X_test, _ = loader.load_split("test", dev_mode=True)
    
    # Ensure length match
    if len(X_test) != len(df_preds):
        logger.warning(f"Length mismatch: X_test={len(X_test)}, Preds={len(df_preds)}. Truncating...")
        X_test = X_test.iloc[:len(df_preds)].reset_index(drop=True)
        
    logger.info("[Step 2/6] Computing Composite Risk Score...")
    scorer = RiskScorer(config)
    df_scored = scorer.compute_risk_scores(df_preds, X_test, df_segments)
    
    logger.info("[Step 3/6] Classifying Risk Levels...")
    classifier = RiskClassifier(config)
    df_classified = classifier.classify(df_scored)
    boundaries = classifier.get_boundaries()
    logger.info(f"  Boundaries: Low/Med = {boundaries['low_threshold']:.1f}, Med/High = {boundaries['medium_threshold']:.1f}")
    
    logger.info("[Step 4/6] Generating Root Causes & Business Recommendations...")
    engine = RecommendationEngine(config)
    df_final = engine.generate_recommendations(df_classified)
    
    # Filter only relevant columns for the final dataset
    final_cols = [
        "item_id", "store_id", "actual", "point", "lower_sym", "upper_sym", 
        "Risk_Score", "Risk_Level", "Root_Cause", "Recommended_Action", 
        "Intervention", "Recommended_Stock_Level", "Inventory_Increase", "Confidence_Level"
    ]
    df_dashboard = df_final[final_cols].copy()
    
    logger.info("[Step 5/6] Generating Visualizations...")
    visualizer = DecisionVisualizer(config["outputs"]["plots_dir"])
    visualizer.plot_risk_distribution(df_final, boundaries, config["plots"]["risk_distribution"])
    visualizer.plot_risk_level_distribution(df_final, config["plots"]["risk_level_dist"])
    visualizer.plot_top_high_risk_items(df_final, config["plots"]["top_10_risk_items"])
    visualizer.plot_risk_heatmap(df_final, config["plots"]["risk_heatmap"])
    
    logger.info("[Step 6/6] Saving Decision Artifacts...")
    dec_dir = Path(config["outputs"]["decision_dir"])
    dec_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(config["outputs"]["reports_dir"])
    
    # Save dashboard dataset
    df_dashboard.to_parquet(dec_dir / config["artifacts"]["dashboard_data"], index=False)
    
    # Save classification report
    counts = df_final["Risk_Level"].value_counts().to_dict()
    report = {
        "risk_level_counts": counts,
        "score_boundaries": boundaries,
        "weights_used": config["decision"]["score_weights"]
    }
    with open(rep_dir / config["artifacts"]["classification_report"], "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("  ✓ " + config["plots"]["risk_distribution"])
    logger.info("  ✓ " + config["plots"]["risk_level_dist"])
    logger.info("  ✓ " + config["plots"]["top_10_risk_items"])
    logger.info("  ✓ " + config["plots"]["risk_heatmap"])
    logger.info("  ✓ " + config["artifacts"]["dashboard_data"])
    logger.info("  ✓ " + config["artifacts"]["classification_report"])
    logger.info("====================================================================")
    logger.info("STAGE 6 COMPLETE ✓")
    logger.info("====================================================================")

if __name__ == "__main__":
    main()
