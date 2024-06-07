import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df =pd.read_pickle("../../data/interim/data_proccessed.pkl")

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df = df[df["Set"] == 1]
plt.plot(set_df["acc_y"])

plt.plot(set_df["acc_y"].reset_index(drop = True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["Label"].unique():
    subset = df[df["Label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop = True), label = label)
    plt.legend()
    plt.show()
    
for label in df["Label"].unique():
    subset = df[df["Label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop = True), label = label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("Label == 'squat'").query("Participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["Category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_df = df.query("Label == 'bench'").sort_values("Participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["Participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("Samples")
plt.legend()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

Label = "squat"
Participant = "A"

all_axix_df = df.query(f"Label == '{Label}'").query(f"Participant == '{Participant}'").reset_index()

fig, ax = plt.subplots()
all_axix_df[["acc_x","acc_y","acc_z"]].plot()
ax.set_ylabel("Acceletor")
ax.set_xlabel("Samples")
plt.legend()



# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

Labels = df["Label"].unique()
Participants = df["Participant"].unique()

for Label in Labels:
    for Participant in Participants:
        all_axix_df = (
            df.query(f"Label == '{Label}'")
            .query(f"Participant == '{Participant}'")
            .reset_index()
        )
        
        if len(all_axix_df) > 0:
            fig, ax = plt.subplots()
            all_axix_df[["acc_x","acc_y","acc_z"]].plot(ax =ax)
            ax.set_ylabel("Acceletor")
            ax.set_xlabel("Samples")
            plt.title(f"{Label} ({Participant})".title())
            plt.legend()

for Label in Labels:
    for Participant in Participants:
        all_axix_df = (
            df.query(f"Label == '{Label}'")
            .query(f"Participant == '{Participant}'")
            .reset_index()
        )
        
        if len(all_axix_df) > 0:
            fig, ax = plt.subplots()
            all_axix_df[["gyr_x","gyr_y","gyr_z"]].plot(ax =ax)
            ax.set_ylabel("Gyroscope")
            ax.set_xlabel("Samples")
            plt.title(f"{Label} ({Participant})".title())
            plt.legend()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


Label = "row"
Participant = "A"

combined_plot_df = df.query(f"Label == '{Label}'").query(f"Participant == '{Participant}'").reset_index(drop=True)

fig, ax = plt.subplots(nrows= 2, sharex= True, figsize=(20,10))
combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax =ax[0])
combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax =ax[1])

ax[0].legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[0].set_ylabel("Acceletor")
ax[1].set_ylabel("Gyroscope")
ax[1].set_xlabel("Samples")
# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------


Labels = df["Label"].unique()
Participants = df["Participant"].unique()

for Label in Labels:
    for Participant in Participants:
        combined_plot_df = ( 
         df.query(f"Label == '{Label}'")
         .query(f"Participant == '{Participant}'")
         .reset_index(drop=True)
         )

        
        
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows= 2, sharex= True, figsize=(20,10))
            combined_plot_df[["acc_x","acc_y","acc_z"]].plot(ax =ax[0])
            combined_plot_df[["gyr_x","gyr_y","gyr_z"]].plot(ax =ax[1])

            ax[0].legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center",bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[0].set_ylabel("Acceletor")
            ax[1].set_ylabel("Gyroscope")
            ax[1].set_xlabel("Samples")
            
            plt.savefig(f"../../reports/figures/{Label.title()}({Participant}).png")
            plt.show()