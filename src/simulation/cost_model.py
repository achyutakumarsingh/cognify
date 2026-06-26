import pandas as pd
from typing import Dict, Any

class CostModel:
    """
    Computes financial consequences of inventory decisions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.holding_rate = config["simulation"]["financial"]["holding_cost_rate"]
        self.stockout_rate = config["simulation"]["financial"]["stockout_penalty_rate"]

    def evaluate_financials(
        self, 
        df: pd.DataFrame, 
        prefix: str, 
        holding_rate: float = None, 
        stockout_rate: float = None
    ) -> pd.DataFrame:
        """
        Computes costs for a given inventory policy (indicated by prefix).
        Prefix can be 'Baseline' or 'Proposed'.
        Requires columns: {prefix}_Order, actual, sell_price
        """
        h_rate = holding_rate if holding_rate is not None else self.holding_rate
        s_rate = stockout_rate if stockout_rate is not None else self.stockout_rate
        
        order_col = f"{prefix}_Order"
        
        # Physical units
        stockout_units = (df["actual"] - df[order_col]).clip(lower=0)
        overstock_units = (df[order_col] - df["actual"]).clip(lower=0)
        sold_units = df[["actual", order_col]].min(axis=1)
        
        # Financial costs
        holding_cost = overstock_units * (df["sell_price"] * h_rate)
        stockout_cost = stockout_units * (df["sell_price"] * s_rate)
        total_cost = holding_cost + stockout_cost
        
        # Store in dataframe
        df[f"{prefix}_Stockout_Units"] = stockout_units
        df[f"{prefix}_Overstock_Units"] = overstock_units
        df[f"{prefix}_Sold_Units"] = sold_units
        df[f"{prefix}_Holding_Cost"] = holding_cost
        df[f"{prefix}_Stockout_Cost"] = stockout_cost
        df[f"{prefix}_Total_Cost"] = total_cost
        
        return df
