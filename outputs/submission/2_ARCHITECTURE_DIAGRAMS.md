# Architecture & System Diagrams

The following professional-grade diagrams illustrate the architecture, data flow, and components of the Supply Chain Decision Intelligence Platform. They are formatted in Mermaid.js, rendering directly in GitHub, Notion, and other modern markdown viewers.

---

## 1. Overall System Architecture

```mermaid
graph TD
    subgraph Data Input Layer
        A1[Raw Sales Data] --> B[Data Loader & Validator]
        A2[Calendar & Pricing] --> B
    end

    subgraph Feature Pipeline
        B --> C[Feature Engineering]
        C --> C1(Temporal Features)
        C --> C2(Rolling Aggregates)
        C --> C3(Sparsity Flags)
    end

    subgraph AI Forecasting Engine
        C --> D[XGBoost Regression]
        D --> E(Tweedie Objective)
        D --> F(Point Forecasts)
    end

    subgraph Uncertainty Quantification Engine
        F --> G{UQ Method}
        C --> G
        G -->|Quantile Regression| H1[Pinball Loss Training]
        G -->|Conformal Prediction| H2[Holdout Residual Calibration]
        H1 --> I[Prediction Intervals]
        H2 --> I
    end

    subgraph Decision & Business Logic
        I --> J[Calibration Evaluator]
        J --> K[Risk Scoring Engine]
        K --> L(Composite Risk Score)
        L --> M[Inventory Simulation Engine]
        M --> N(Financial & Service Level KPIs)
    end

    subgraph Presentation Layer
        N --> O[Streamlit Executive Dashboard]
        L --> O
        F --> O
        I --> O
    end

    classDef core fill:#1e1b4b,stroke:#4f46e5,stroke-width:2px,color:#e0e7ff;
    classDef highlight fill:#4f46e5,stroke:#a5b4fc,stroke-width:2px,color:#ffffff;
    classDef sub fill:#312e81,stroke:#6366f1,stroke-width:1px,color:#c7d2fe;
    
    class A1,A2,B,C,D,G,I,J,K,M,O core;
    class O highlight;
    class C1,C2,C3,E,F,H1,H2,L,N sub;
```

---

## 2. Supply Chain Decision Flow

```mermaid
sequenceDiagram
    participant P as Supply Chain Planner
    participant D as Streamlit Dashboard
    participant R as Risk Engine
    participant S as Simulation Engine
    participant M as ML Pipeline

    P->>D: Requests Inventory Action for SKU-123
    D->>M: Fetch Point Forecast
    M-->>D: Returns "Predicted: 50 units"
    D->>M: Fetch Uncertainty Interval
    M-->>D: Returns "90% CI: [20, 85] units"
    
    D->>R: Request Risk Triage
    R-->>D: Returns Risk Level: HIGH (Score: 82/100)
    
    D->>S: Run Financial Impact Simulation
    S-->>D: Returns Recommended Action (Order 85 units)
    
    D->>P: Display Plain-English Explanation & ROI
    P->>D: Adjusts Confidence Slider (90% -> 95%)
    D->>S: Re-run Simulation with new Alpha
    S-->>D: Returns Updated Costs
    D->>P: Displays New Recommendation
```

---

## 3. Risk Engine Scoring Mechanism

```mermaid
flowchart LR
    A[Point Forecast] --> D
    B[Prediction Interval] --> D
    C[Historical Errors] --> D
    
    D[Risk Engine] --> E{Composite Score}
    
    E -->|Interval Width > Threshold| F[High Width Penalty]
    E -->|High Error Variance| G[Forecast Error Penalty]
    E -->|High Demand Standard Deviation| H[Volatility Penalty]
    E -->|Poor Empirical Coverage| I[Calibration Penalty]
    
    F --> J((Final Score 0-100))
    G --> J
    H --> J
    I --> J
    
    J --> K[LOW RISK]
    J --> L[MEDIUM RISK]
    J --> M[HIGH RISK]

    classDef green fill:#14532d,stroke:#22c55e,color:#fff;
    classDef yellow fill:#78350f,stroke:#f59e0b,color:#fff;
    classDef red fill:#7f1d1d,stroke:#ef4444,color:#fff;
    
    class K green;
    class L yellow;
    class M red;
```

---

## 4. Dashboard Architecture & Routing

```mermaid
graph TD
    A[app.py Entry Point] --> B[data_loader.py @st.cache_data]
    B --> C[(Outputs: Parquet & JSON)]
    
    A --> D{Sidebar Navigation}
    
    D --> P1[1. Executive Overview]
    D --> P2[2. Demand Forecasting]
    D --> P3[3. Uncertainty Analysis]
    D --> P4[4. Calibration Dashboard]
    D --> P5[5. Risk Intelligence]
    D --> P6[6. Business Impact]
    D --> P7[7. Scenario Simulator]
    D --> P8[8. Product Drill-Down]
    D --> P9[9. Explainability]
    D --> P10[10. Demo Mode]
    
    P1 -.-> U[components/ui.py]
    P2 -.-> U
    P5 -.-> U
```
