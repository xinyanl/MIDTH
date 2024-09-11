import pandas as pd
import statsmodels.api as sm
from kneed import KneeLocator
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import STL

### 1. get battery polarization elbow point
def get_elbow_point(charge_agg_data):
    """
    Takes a polarization curve,
    applies smoothing and locate the polarization elbow point,
    returns the elbow point 
    
    Parameters:
        charge_agg_data: pd.DataFrame, the polarization curve        

    Returns:
        elbow: int, the elbow point on the charge polarization curve
        ax: matplotlib.figure.Figure,
            a plot showing the identification of elbow points
    """
    fig, ax = plt.subplots()
    charge_agg_data.plot("cycle_number", "U_median", ax=ax, label="org",
        marker="o",
        markersize=9,
        markeredgewidth=2,
        linewidth=1)
    s = pd.Series(
        sm.nonparametric.lowess(
            charge_agg_data["U_median"], 
            charge_agg_data["cycle_number"], 
            frac=0.25
        )[:, 1]
    )
    s.plot(ax=ax, label="smoothed",
           marker="o",
        markersize=5,
        markeredgewidth=2,
        linewidth=1)
    elbow = KneeLocator(
        s.iloc[:].index, 
        s.iloc[:].values, 
        S=1, curve="convex", direction="increasing", interp_method="interp1d"
    ).elbow
    ax.axvline(elbow, color="r", label="elbow")
    ax.legend()
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("U_median_charge")
    return elbow, ax

### 2.1. Curve segmentation
def _get_sep_pt_single_cycle(one_cycle_df):
    """
    Takes a charge curve,
    locates the elbow knee/elbow points, 
    returns all the knee/elbow points of a single cycle
    
    Parameters:
        one_cycle_df: pd.DataFrame, the charging curve of a single cycle        

    Returns:
        pt_se: pd.Series, all the locations of the knee/elbow points on the charging curve
    """
    #Get the 2nd knee point
    vals = one_cycle_df.reset_index(drop=True)["U"]
    cut_pt = int(0.4*len(vals))
    tempt_kneedle = KneeLocator(
        vals.iloc[cut_pt:].index, vals.iloc[cut_pt:].values, 
        S=1, curve="convex", direction="increasing", interp_method="interp1d"
    )
    delta_s = 15
    delta_e = 6
    offset = int(0.2 * len(tempt_kneedle.y_difference))
    maxima = np.argmax(tempt_kneedle.y_difference[offset:]) + offset
    start = max(len(tempt_kneedle.y_difference) - maxima - delta_s + cut_pt, cut_pt)
    end = len(tempt_kneedle.y_difference) - maxima + delta_e + cut_pt
    kneedle = KneeLocator(
        vals.iloc[start:end].index, 
        vals.iloc[start:end].values, 
        S=1.e-3, curve="convex", direction="increasing", interp_method="interp1d",
        online=True,
    )
    e2 = kneedle.elbow
    pt2 = 1.*e2/len(vals) if e2 else np.nan
    #Get the 1st knee point
    start = 0       
    end = int(0.5*len(vals))
    kneedle = KneeLocator(
        vals.iloc[start:end].index, 
        vals.iloc[start:end].values, 
        S=1, curve="convex", direction="decreasing", interp_method="polynomial",
        online=True
    )
    e1 = kneedle.elbow
    pt1 = 1.*e1/len(vals)
    pt_se = pd.Series(
        [pt1, pt2, pt1*len(vals), pt2*len(vals)], 
        ["del_pt1", "del_pt2", "ab_pt1", "ab_pt2"]
    )
    return pt_se

def get_sep_pt_all_cycles(df):
    """
    Takes charging curves for all cycles,
    applies the turning point identification method to each cycle,
    returns the turning points for each cycle
    
    Parameters:
        df: pd.DataFrame, the charging curve for a single cycle        

    Returns:
        t: pd.Series, the locations of all knee/elbow points on the charging curve
    """
    t = df.groupby(
            ["cycle_number"]
        ).apply(_get_sep_pt_single_cycle).reset_index()
    t["pt1"] = t["del_pt1"] + t["cycle_number"]
    t["pt2"] = t["del_pt2"] + t["cycle_number"]
    t["del_pt12"] = t["pt2"] - t["pt1"]
    return t

def plot_charge_curve_segments(EP, df, sep_pt_df):
    """
    Takes the charging curves and their turning points,
    plots the method for segmenting the charging curves
    
    Parameters:
        EP: int, the elbow point
        df: pd.DataFrame, the charge curve data
        sep_pt_df: pd.DataFrame, the knee/elbow points for all cycles
        
    Returns:
        ax: matplotlib.figure.Figure, the figure illustrating the curve segmentation
    """
    fig, ax = plt.subplots()
    cycles = np.arange(1, EP+1, 2)
    colors = plt.cm.Blues(np.linspace(0, 1, len(cycles)))
    for m, n in enumerate(cycles):
        tempt = df.query(
            "cycle_number==%d" %n 
        ).copy(deep=True)
        tempt["norm_q"] = tempt["tot_capacity"] / tempt["tot_capacity"].max()
        ax.plot(
            tempt["norm_q"],
            tempt["U"],
             color=colors[m]
        )
        e1, e2 = sep_pt_df.query("cycle_number==%d" %n).iloc[0][
            ["del_pt1", "del_pt2"]
        ]
        ax.axvline(e1, color=colors[m], linestyle="-")
        ax.axvline(e2, color=colors[m], linestyle="-")
    ax.set_xlim((0, 1))
    ax.set_yticks(np.linspace(0.0, 0.3, 4))
    ax.set_xlabel("Normalized capacity", fontsize=20)
    ax.set_ylabel("Voltage (V)", fontsize=20)
    return ax

### 2.2. Curve decomposition
def _normalize_season(df, first_cycle, cols=["seasonality"]):
    df = df.reset_index(drop=True)
    df = df.join(first_cycle[cols], rsuffix="_first")
    for col in cols:
        df["%s_norm" %col] = df[col] - df["%s_first" %col] 
    return df[
        ["charge_cycle", "charge_life", "U", "U_median", "trend", "resid"] 
        + cols + ["%s_norm" %col for col in cols]
    ]

def _ts_stl_decompose(
    ts, trend_smoothing, season_smoothing, period, log=False
):
    """
    decompose the charging curves into trend and seasonality parts
    """
    if log:
        ts = ts.apply(np.log)
    res = STL(
        ts, period=period, trend=trend_smoothing, seasonal=season_smoothing
    ).fit()    
    observed = res.observed.apply(np.exp) if log else res.observed
    trend = res.trend.apply(np.exp) if log else res.trend
    seasonal = res.seasonal.apply(np.exp) if log else res.seasonal
    resid = res.resid.apply(np.exp) if log else res.resid
    res_df = observed.reset_index().join(
        trend.reset_index()[["trend"]]
    ).join(
        seasonal.reset_index()[["season"]]
    ).join(
        resid.reset_index()[["resid"]]
    )
    res_df.columns = ["charge_cycle", "U", "trend", "seasonality", "resid"]
    res_df["cycle_number"] = pd.Series(res_df.index/period).apply(int)
    return res_df

### 3. LSTM_training
class VarImpVIANN(Callback):
    """
    VarImpVIANN aims to determine the importance of each input feature (variable) 
    on the model's prediction outcomes
    """ 
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.n = 0
        self.M2 = 0.0
        self.layer = 1

    def on_train_begin(self, logs={}, verbose = 1):
        if self.verbose:
            print("VIANN version 1.0 (Wellford + Mean) update per epoch")
        self.diff = self.model.layers[self.layer].get_weights()[0]
        
    def on_epoch_end(self, batch, logs={}):
        currentWeights = self.model.layers[self.layer].get_weights()[0]
        self.n += 1
        delta = np.subtract(currentWeights, self.diff)
        self.diff += delta/self.n
        delta2 = np.subtract(currentWeights, self.diff)
        self.M2 += delta*delta2
        self.lastweights = self.model.layers[self.layer].get_weights()[0]

    def on_train_end(self, batch, logs={}):
        if self.n < 2:
            self.s2 = float("nan")
        else:
            self.s2 = self.M2 / (self.n - 1)
        scores = np.sum(np.multiply(self.s2, np.abs(self.lastweights)), axis = 1)
        self.varScores = (scores - min(scores)) / (max(scores) - min(scores))
        important_indices = np.array(self.varScores).argsort()[-10:][::-1]
        if self.verbose:
            print("Most important variables: ",
                  important_indices)
            for idx in important_indices:
                print(f"Variable {idx} importance: {self.varScores[idx]}")
        important_values = [(idx, self.varScores[idx]) for idx in important_indices]
        np.savetxt("feature_impt.csv", important_values, delimiter=",", header="Variable Index,Importance", comments='')  
    
### 4. feature importance ranking
def feature_importance_ranking(feature_impt):
    """
    Takes the feature importance values,
    plots the method for segmenting the charging curves
    
    Parameters:
        feature_impt: list, the importance values of all features
        
    Returns:
        ax: matplotlib.figure.Figure, the figure illustrating the feature ranking
    """
    impt = np.array(feature_impt)
    sorted_idx = np.argsort(impt)
    fig, ax = plt.subplots()
    pos = np.arange(impt.shape[0])
    label_cols = [
        "T$_{2}$", "P$_{1}$", "P$_{2}$", "P$_{2}$–P$_{1}$", "T$_{1}$",
        "T$_{2}$–T$_{3}$", "T$_{1}$–T$_{2}$", "S$_{1}$", "S$_{3}$",
    ]
    ax.barh(
        pos[-1], impt[sorted_idx[-1]], align="center", alpha=0.8,
    )
    ax.barh(
        pos[-2], impt[sorted_idx[-2]], align="center", alpha=0.8,
    )
    ax.barh(
        pos[:-2], impt[sorted_idx[:-2]], align="center", alpha=0.8,
    )
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(label_cols)[sorted_idx])
    ax.set_xlabel("Normalized feature importance", fontsize=20)
    return ax
