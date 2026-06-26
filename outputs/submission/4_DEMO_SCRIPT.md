# 3-Minute Demo Script: Supply Chain Decision Intelligence

## Setup (Before Demo)
- Ensure the app is running via `streamlit run app.py` (locally at `http://localhost:8501` or on Streamlit Cloud at `https://cognifyaiprediction.streamlit.app/`).
- Have the app open full-screen in a clean browser window.
- Ensure the sidebar is expanded.
- Make sure **Guided Demo Mode** is *unchecked* to start.

---

## [0:00 - 0:45] Step 1: Introduction & Executive Overview
*(Action: Check the **🚀 Guided Demo Mode** box in the sidebar)*

**Speaker:**
> "Hello judges. Welcome to our Supply Chain Decision Intelligence platform. Traditional AI supply chain systems give managers a single, deterministic number—a point forecast. But if you don't know the *confidence* of that forecast, you are forced to guess safety stock, causing either costly stockouts or bloated inventory."

> "To solve this, we built a Decision Intelligence tool. Looking at the **Executive Overview**, you can see we monitor over 3,000 products. The AI has flagged 838 high-risk items requiring attention. By implementing our risk-aware safety stock policy instead of baseline point forecasting, we reduce total operating costs by 15.6% and increase service levels to 92.2%."

---

## [0:45 - 1:30] Step 2: Risk & Operations (The Workflow)
*(Action: Click the **Next Step ➡️** button on the top banner)*

**Speaker:**
> "Now, we step into the **Risk & Operations** center. Planners shouldn't have to analyze math models. We consolidate product data into a simple risk triage table. Let's look at a specific item."

*(Action: Click the 'Single Product Analysis' tab and search for `FOODS_1_001` or look at the first default product)*
> "Here, we plot the actual demand against our XGBoost forecast, but we wrap it in a statistically calibrated 90% Conformal Prediction interval. The pink dashed line is our recommended safety stock, scaling up automatically during periods of high demand uncertainty."

*(Action: Scroll down and click the **⚙️ Send PO to ERP (Demo)** button)*
> "Most importantly, this is a workflow tool. With one click, the planner can approve the PO recommendations and automatically dispatch the modified safety stock targets directly to SAP or Oracle Cloud ERP via simulated REST API triggers."

---

## [1:30 - 2:15] Step 3: Scenario Simulator (Interactive Sandbox)
*(Action: Click the **Next Step ➡️** button on the top banner)*

**Speaker:**
> "But what happens if lead times double or supply chain volatility spikes? In our **Scenario Simulator**, planners have a real-time sandbox."

*(Action: Slide the 'Replenishment Lead Time' from 1 day to 5 days, and then slide 'Demand Volatility Multiplier' to 1.3x)*
> "As I adjust these sliders, our in-memory engine uses supply chain physics—scaling uncertainty bounds by the square root of lead time and volatility—to immediately recalculate and display the new expected safety stock holding costs, stockout probabilities, and net savings. Planners can see the exact break-even point for deploying the AI policy."

---

## [2:15 - 3:00] Step 4: Technical Engine & Close
*(Action: Click the **Next Step ➡️** button on the top banner)*

**Speaker:**
> "Finally, for the technical judges, our **Technical Engine** exposes the statistical integrity of our models."

*(Action: Briefly click through the 'Point Forecast & Residuals', 'Uncertainty Bounds', and 'Calibration & Reliability' tabs)*
> "We validate the point forecast residuals, compare Split Conformal Prediction against Quantile Regression, and plot our reliability diagrams. Our Conformal bounds achieve ~87.5% empirical coverage, showing our uncertainty intervals are statistically calibrated and reliable."

*(Action: Click **Finish Demo 🔁**)*
> "By bridging the gap between advanced statistics and daily logistics workflows, we've turned raw model outputs into a 15.6% cost reduction. The system is live, scalable, and ready for deployment. Thank you, and we'd love to take your questions."
