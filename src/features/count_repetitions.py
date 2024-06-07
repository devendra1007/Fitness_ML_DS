import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df =pd.read_pickle("../../data/interim/data_proccessed.pkl")
df = df[df["Label"] != "rest"]

acc_r =df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r =df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2

df["acc_r"]=np.sqrt(acc_r)
df["gyr_r"]=np.sqrt(gyr_r)


# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["Label"] == "bench"]
squat_df = df[df["Label"] == "squat"]
row_df = df[df["Label"] == "row"]
ohp_df = df[df["Label"] == "ohp"]
dead_df = df[df["Label"] == "dead"]


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df

plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["Set"] == plot_df["Set"].unique()[0]]["gyr_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs =1000/200
LowPass=LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["Set"] == bench_df["Set"].unique()[0]] 
squat_set = squat_df[squat_df["Set"] == squat_df["Set"].unique()[0]] 
row_set = row_df[row_df["Set"] == row_df["Set"].unique()[0]] 
ohp_set = ohp_df[ohp_df["Set"] == ohp_df["Set"].unique()[0]] 
dead_set = dead_df[dead_df["Set"] == dead_df["Set"].unique()[0]] 

bench_set["acc_r"].plot()
col = "acc_r"
LowPass.low_pass_filter(bench_set,col=col,sampling_frequency=fs,cutoff_frequency=0.4,order=10)[col + "_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

col = "acc_r"

data = LowPass.low_pass_filter(bench_set,col=col,sampling_frequency=fs,cutoff_frequency=0.4,order=10)
argrelextrema(data[col + "_lowpass"].values,np.greater)


def count_rep(dataset, cutoff =0.4,order =10,col = "acc_r"):
    data = LowPass.low_pass_filter(dataset,col=col,sampling_frequency=fs,cutoff_frequency=cutoff,order=order)
    indexes = argrelextrema(data[col + "_lowpass"].values,np.greater)
    peaks = data.iloc[indexes]
    
    fig,ax = plt.subplots()
    plt.plot(dataset[f"{col}_lowpass"])
    plt.plot(peaks[f"{col}_lowpass"],"o",color ="red")
    ax.set_ylabel([f"{col}_lowpass"])
    exc=dataset["Label"].iloc[0].title()
    cat = dataset["Category"].iloc[0].title()
    plt.title(f"{cat} {exc} Set: {len(peaks)} Reps")
    plt.show()
    
    return len(peaks)

count_rep(bench_set,cutoff=0.4)
count_rep(squat_set,cutoff=0.35)
count_rep(row_set,cutoff=0.65,col="gyr_x")
count_rep(ohp_set,cutoff=0.35)
count_rep(dead_set,cutoff=0.4)
    
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["Category"].apply(lambda x: 5 if x == "heavy" else 10)
rep_df = df.groupby(["Label", "Category","Set"])["reps"].max().reset_index()

rep_df["reps_pred"] = 0

for s in df["Set"].unique():
    subset = df[df["Set"] == s]
    
    col="acc_r"
    cutoff = 0.4
    
    if subset["Label"].iloc[0] == "squat":
        cutoff = 0.35
    if subset["Label"].iloc[0] == "row":
        cutoff = 0.65
        col = "gyr_x"
    if subset["Label"].iloc[0] == "ohp":
        cutoff = 0.35
        
    reps = count_rep(subset,cutoff=cutoff,col=col)
    rep_df.loc[rep_df["Set"] == s, "reps_pred"] = reps
    
rep_df

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"],rep_df["reps_pred"]).round(2)
rep_df.groupby(["Label","Category"])["reps","reps_pred"].mean().plot.bar()