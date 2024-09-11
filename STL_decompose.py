from helper_functions import _ts_stl_decompose, _normalize_season
import numpy as np
def decomp_and_process_df(ts, smoothing_params, period):
    """
    Takes a charge curve time series and parameters to be used,
    returns the decomposed time series and the aggregated results by cycle number

    Parameters:
        ts: pd.Series, the charge voltage curve
        smoothing_params: tuple, (trend_smoothing, season_smoothing), 
                          the smoothing parameters to be used for STL decomposition
        period: int, the periodicity of the time series
                          
    Returns:
        decomp_df: pd.DataFrame, the un-aggregated decomposed charge curve, 
                   components include the original charge curve and the decomposed
                   trend, seasonality and residual
        df_agg: pd.DataFrame, the results aggregated on cycle number level, 
                resulting features will be used in model training
    """
    decomp_df = _ts_stl_decompose(ts, smoothing_params[0], smoothing_params[1], period)
    first_cycle = decomp_df.query("cycle_number==0").reset_index(drop=True)
    df_norm = decomp_df.groupby(["cycle_number"]).apply(
        _normalize_season, first_cycle
    ).reset_index(drop=True)
    df_agg = df_norm.groupby(["cycle_number"]).agg({
        "trend": "mean",
        "seasonality_norm": lambda x: x.apply(np.abs).sum(),
    })
    df_agg.columns = ["trend_mean", "seasonality_norm_area"]
    return  decomp_df, df_agg.reset_index()
