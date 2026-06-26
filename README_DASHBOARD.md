# Supply Chain Decision Intelligence Dashboard

## Overview
This is the final production dashboard for the **AI-Powered Supply Chain Decision Intelligence System**. It presents the complete 8-stage ML pipeline — from raw demand data to actionable inventory decisions — in a polished, interactive interface suitable for live hackathon demonstration.

## Launch Instructions
```bash
# From the project root directory:
streamlit run app.py
```
The dashboard will open at **http://localhost:8501**.

## Pages
| Page | Description |
|---|---|
| 🏢 Executive Overview | KPI cards, risk distribution, cost comparison |
| 📈 Demand Forecasting | Actual vs forecast, residuals, feature importance |
| 🎯 Uncertainty Analysis | Conformal vs Quantile intervals, confidence slider |
| 📐 Calibration Dashboard | Reliability diagrams, coverage by segment |
| ⚠️ Risk Intelligence | Risk scores, root causes, sortable risk table |
| 💰 Business Impact | KPI comparison, financial simulation, scenario results |
| 🔬 Scenario Simulator | Interactive cost/confidence parameter adjustment |
| 🔍 Product Drill-Down | Per-product forecast + risk + recommendation |
| 💬 Explainability | Plain-English explanation of every recommendation |
| ▶️ Demo Mode | Guided 3-minute end-to-end demo for judges |

## Requirements
```bash
pip3 install -r requirements.txt
```

## Project Structure
```
MLProj/
├── app.py                    # Streamlit entry point
├── dashboard/
│   ├── data_loader.py        # Cached data access layer
│   ├── components/
│   │   └── ui.py             # Reusable UI primitives
│   └── pages/
│       ├── p1_executive_overview.py
│       ├── p2_demand_forecasting.py
│       ├── p3_uncertainty_analysis.py
│       ├── p4_calibration_dashboard.py
│       ├── p5_risk_intelligence.py
│       ├── p6_business_impact.py
│       ├── p7_scenario_simulator.py
│       ├── p8_product_drilldown.py
│       ├── p9_explainability.py
│       └── p10_demo_mode.py
├── src/                      # ML pipeline source code (Stages 1-7)
├── config/                   # YAML configuration files
└── outputs/                  # Generated artifacts and reports
```

## Demo Instructions (For Judges)
1. Launch the app: `streamlit run app.py`
2. Navigate to **▶️ Demo Mode** in the sidebar
3. Click **▶ Run Demo** for a guided 3-minute walkthrough
4. Use the sidebar to explore any specific page in detail
