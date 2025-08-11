# -*- coding: utf-8 -*-
"""
Kyle Mellors
MSDS 670 - Week 5 Assignment
"""

#%%
# import libraries and data

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dpi = 300

# if you are on windows, you are going to have to change the file path to
# windows style
project_dir = r'C:\Users\kmell\Regis_MSDS\MSDS_670_Data_Visualization\MSDS_670_Wk5/'
data_dir = project_dir + r'data/'
output_dir = project_dir + r'output/'

df_filename = 'SalesCallData.csv'
df = pd.read_csv(data_dir + df_filename)
columns = list(df.columns)

#%%
# EDA
for col in ["Branch", "Call Purpose", "Incoming or Outgoing", "Queue", "Rep ID", "Sale", "Shift"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

#%%
# Visualization 1: Evening Shifts per Representative (Horizontal Bar)
mask = df["Shift"].astype(str).str.strip().str.casefold() == "evening"
evening_counts = (
    df.loc[mask]
      .groupby("Rep ID")
      .size()
      .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(evening_counts.index, evening_counts.values, color="teal")
ax.set_title("Evening Shifts per Representative")
ax.set_xlabel("Number of Evening Shifts")
ax.set_ylabel("Rep ID")

fig.tight_layout()
fig.savefig(output_dir + "rep_evening_shifts_barh.png", dpi=dpi, bbox_inches="tight")
plt.show()
#%%
# Visualization 2: Distribution of Waiting Minutes (Horizontal Bar w/ Bins)
if "Waiting Minutes" in df.columns:
    wm = df["Waiting Minutes"].dropna().astype(float)
    if not wm.empty:
        low, high = int(wm.min()), int(wm.max())
        bins = range(low, high + 2)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(wm.values, bins=bins, align="left", rwidth=0.9, color="slateblue")
        ax.set_title("Distribution of Waiting Minutes")
        ax.set_xlabel("Wait Times (in minutes)")
        ax.set_ylabel("Number of Calls")
        fig.tight_layout()
        fig.savefig(output_dir + "hist_waiting_minutes.png", dpi=dpi, bbox_inches="tight")
        plt.show()
#%%
# Visualization 3: Total Crimes per Year (Single Line Chart)
df2_filename = 'Denver Crime 2001-2013.csv'
df2 = pd.read_csv(data_dir + df2_filename)
columns = list(df2.columns)

years = sorted([int(c) for c in df2.columns if c.isdigit()])
year_cols = [str(y) for y in years]


totals = df2[year_cols].sum(axis=0)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years, totals.values, marker="o")
ax.set_title("Total Reported Crimes by Year 2001-2013 (Denver)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crimes")

fig.tight_layout()
fig.savefig(output_dir + "denver_total_crimes_by_year.png", dpi=dpi, bbox_inches="tight")
plt.show()
#%%
# Visualization 4: Number of Crimes per Type in Denver (multi-line)
wide = df2.set_index("Type")[year_cols]
wide = wide.loc[wide.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_prop_cycle(color=plt.cm.tab10(np.linspace(0, 1, min(len(wide), 9))))

for typ, row in wide.iterrows():
    ax.plot(years, row.values, marker="o", label=str(typ))

ax.set_title("Number of Crimes per Crime Type in Denver (2001-2013)")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Crimes")
ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")

fig.tight_layout()
fig.savefig(output_dir + "denver_trends_by_type.png", dpi=dpi, bbox_inches="tight")
plt.show()
