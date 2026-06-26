# 3-Minute Demo Script: Supply Chain Decision Intelligence

## Setup (Before Demo)
- Ensure the app is running via `streamlit run app.py` at `http://localhost:8501`.
- Have the app open full-screen in a clean browser window.
- Ensure the sidebar is expanded.

---

## [0:00 - 0:30] Introduction & The Problem
*(Screen: Start on **Page 1: Executive Overview**)*

**Speaker:**
> "Hello judges. Welcome to our Supply Chain Decision Intelligence platform. Traditional AI supply chain systems give managers a single, deterministic number—a point forecast. But if you don't know the *confidence* of that forecast, you are forced to guess your safety stock. Today, we're going to show you how we solved this by translating statistical uncertainty into automated financial decisions."

> "Here on the Executive Overview, you can see our system is currently monitoring over 3,000 products, and our Risk Engine has immediately flagged 838 of them as High Risk requiring planner attention."

---

## [0:30 - 1:00] Forecasting & Uncertainty
*(Screen: Click **Page 2: Demand Forecasting**)*

**Speaker:**
> "Under the hood, we trained a highly optimized XGBoost engine to handle the zero-inflated retail sales data. But we didn't stop at point forecasts."

*(Screen: Click **Page 3: Uncertainty Analysis**)*

**Speaker:**
> "To prevent planners from guessing, we wrapped every single XGBoost prediction in a statistically guaranteed confidence interval using Split Conformal Prediction."
*(Action: Slowly drag the Confidence Slider from 50% to 90%)*
> "As you can see, our system doesn't require retraining to change risk tolerances. A manager can dynamically adjust their confidence target, and the conformal intervals immediately recalculate on the fly."

---

## [1:00 - 1:30] Risk Triage
*(Screen: Click **Page 5: Risk Intelligence**)*

**Speaker:**
> "We don't expect supply chain managers to analyze probability distributions. So, we built a Risk Engine that absorbs the model's uncertainty, historical errors, and volatility, outputting a composite Risk Score from 0 to 100."
*(Action: Sort the table by "Risk Score" descending)*
> "The system automatically segments the entire catalog, telling the planner exactly which items are High Risk and explicitly what action to take—such as increasing safety stock to the 95th percentile."

---

## [1:30 - 2:00] Business Impact Simulation
*(Screen: Click **Page 6: Business Impact**)*

**Speaker:**
> "But does this actually save money? Yes. We built a financial simulator comparing a baseline strategy—ordering exactly what the model predicts—against our Risk-Aware system."
*(Action: Highlight the Net Savings KPI card)*
> "By dynamically shifting safety stock only to the items with high uncertainty, we reduced total operating costs by 15.6%, eliminated over 1,500 stockout units, and bumped service levels to over 92%."

---

## [2:00 - 2:30] Live Scenario & Explainability
*(Screen: Click **Page 7: Scenario Simulator**)*

**Speaker:**
> "Supply chain economics change rapidly. If warehouse holding costs suddenly double, planners can adjust this slider right here..."
*(Action: Drag the Holding Cost Rate slider higher)*
> "...and the system instantly recalculates the optimal buffer bounds and financial ROI across the entire network."

*(Screen: Click **Page 9: Explainability**)*

**Speaker:**
> "Finally, we know trust is the biggest barrier to AI adoption. For every recommendation, our system generates a plain-English explanation of exactly *why* the AI chose that risk level and *how* the uncertainty components influenced the decision."

---

## [2:30 - 3:00] Automated Demo & Close
*(Screen: Click **Page 10: Demo Mode**)*

**Speaker:**
> "If you'd like to see the entire pipeline summarized, we've built an automated demo mode directly into the application."
*(Action: Click "▶ Run Demo" and let the UI progress bars run)*
> "In summary, we've bridged the gap between pure Data Science and Operations. We turned raw probabilistic bounds into a 15.6% cost reduction, packaged perfectly for enterprise deployment. Thank you."
