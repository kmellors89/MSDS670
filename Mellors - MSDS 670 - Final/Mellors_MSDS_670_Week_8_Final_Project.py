# -*- coding: utf-8 -*-
"""
Kyle Mellors
MSDS 670 - Week 8 Assignment - Final Project
"""

#%%
# import libraries and data

import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import networkx as nx
from collections import Counter
from itertools import combinations
import re

dpi = 300

#%%
project_dir = r'C:\Users\kmell\Regis_MSDS\MSDS_670_Data_Visualization\MSDS_670_Wk8/'
data_dir = project_dir + r'data/'
output_dir = project_dir + r'output/'

df_filename = 'horror_movies.csv'
df = pd.read_csv(data_dir + df_filename)
columns = list(df.columns)
#%%
df.info()
df = df.drop(["overview", 'tagline', 'poster_path', 'adult', 'backdrop_path','collection','collection_name'], axis=1)
df.info()
print(df.head())
#%%
# Visualization 1: Horror Through the Decades (Releases per Decade, 1950s-2010s, Bar Graph)

years = (pd.to_datetime(df["release_date"], errors="coerce")
           .dt.year
           .dropna()
           .astype(int))

years = years[(years >= 1900) & (years <= 2019)]

decades = (years // 10) * 10

dec_min, dec_max = decades.min(), decades.max()
all_decades = np.arange(dec_min, dec_max + 10, 10)
counts = (decades.value_counts()
                  .reindex(all_decades, fill_value=0)
                  .sort_index())

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(counts.index.astype(str) + "s", counts.values, color="#CC5500")
ax.yaxis.set_visible(False)


ax.set_title("Horror Through the Decades: Number of Releases")
ax.set_xlabel("Decade")
ax.set_ylabel("Number of Horror Films Released")

for x, y in zip(range(len(counts)), counts.values):
    ax.text(x, y, str(y), ha="center", va="bottom", fontsize=9)

ax.set_ylim(0, max(counts.values) * 1.10)
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()
#%%
# Visualization 2: Ratings over the Decades
df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

use = df[(df["year"].between(1900, 2019)) & (df["vote_average"].between(0, 10))].copy()

use["decade"] = (use["year"] // 10) * 10

avg_ratings = use.groupby("decade")["vote_average"].mean()

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(avg_ratings.index.astype(str) + "s", avg_ratings.values,
        marker="o", linewidth=2.5, color="#CC5500")

ax.set_title("Average Horror Ratings by Decade")
ax.set_xlabel("Decade")
ax.set_ylabel("Average Rating (0–10)")

for x, y in zip(avg_ratings.index.astype(str) + "s", avg_ratings.values):
    ax.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_ylim(0, 10)
fig.tight_layout()
plt.show()
#%% 
# Visualization 3: Best Horror Movies of all Time (Splatter Plot)
MIN_VOTES = 5000   
TOP_N = 15

use = df[(df["vote_average"].between(0, 10)) & (df["vote_count"] >= MIN_VOTES)].copy()


C = use["vote_average"].mean()
m = int(np.percentile(use["vote_count"], 80))  # votes threshold
use["wr"] = (use["vote_count"] / (use["vote_count"] + m)) * use["vote_average"] + \
            (m / (use["vote_count"] + m)) * C

top = use.nlargest(TOP_N, "wr").copy()

fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(
    top["vote_count"],
    top["vote_average"],
    s=70, alpha=0.85, edgecolors="none", color="#CC5500")

top["year"] = pd.to_datetime(top["release_date"], errors="coerce").dt.year

for _, row in top.iterrows():
    label = f"{row['title'][:25]} ({int(row['year'])})" if not pd.isna(row["year"]) else row["title"][:25]
    ax.annotate(
        label,
        (row["vote_count"], row["vote_average"]),
        xytext=(6, 0),                  # horizontal offset
        textcoords="offset points",
        fontsize=10,
        ha="left", va="center")

ax.set_title("All-Time Best Horror (Top 15): Ratings vs. Vote Count")
ax.set_xlabel("Vote Count")
ax.set_ylabel("Average Rating (0–10)")
ax.set_ylim(6, 9)

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))

fig.tight_layout()
plt.show()
#%%
# Visualization 4: Popular Genres within Horror (Network Graph)

TOP_GENRES = 10         
EDGE_KEEP_Q = 0.40       
NODE_SIZE_RANGE = (1200, 6500)
EDGE_WIDTH_RANGE = (0.8, 5.0)
CUT_AT_2019 = True      
SEED = 4

data = df.copy()
if CUT_AT_2019 and "release_date" in data:
    yrs = pd.to_datetime(data["release_date"], errors="coerce").dt.year
    data = data.loc[yrs.le(2019)]

def parse_genres(s: str):
    if pd.isna(s): return []
    parts = [p.strip() for p in str(s).split(",")]
    parts = [re.sub(r'^[\s"\']+|[\s"\']+$', "", p) for p in parts]
    return [p for p in parts if p]

glist = data["genre_names"].fillna("").apply(parse_genres)

glist = glist[glist.apply(lambda gs: "Horror" in gs and len(gs) > 1)]

genre_counts = Counter()
for gs in glist:
    genre_counts.update(set(gs))   

top_nodes = ["Horror"]
top_nodes += [g for g, _ in genre_counts.most_common(TOP_GENRES*2) if g != "Horror"]
top_nodes = list(dict.fromkeys(top_nodes))[:TOP_GENRES]

edge_counts = Counter()
for gs in glist:
    sel = sorted(set(g for g in gs if g in top_nodes))
    for a, b in combinations(sel, 2):
        edge_counts[(a, b)] += 1

if edge_counts:
    weights = np.array(list(edge_counts.values()), dtype=float)
    thr = np.quantile(weights, EDGE_KEEP_Q)
    edge_counts = Counter({e:w for e, w in edge_counts.items() if w >= thr})

G = nx.Graph()
G.add_nodes_from(top_nodes)
for n in G.nodes:
    G.nodes[n]["count"] = genre_counts.get(n, 0)

for (a, b), w in edge_counts.items():
    G.add_edge(a, b, weight=w)

counts = np.array([G.nodes[n]["count"] for n in G.nodes], dtype=float)
cmin, cmax = counts.min(), counts.max()
ns_min, ns_max = NODE_SIZE_RANGE
if cmax > cmin:
    node_sizes = ns_min + (counts - cmin) / (cmax - cmin) * (ns_max - ns_min)
else:
    node_sizes = np.full_like(counts, (ns_min + ns_max) / 2)

eweights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float) if G.edges else np.array([])
ew_min, ew_max = EDGE_WIDTH_RANGE
if eweights.size:
    ew = ew_min + (eweights - eweights.min()) / (eweights.ptp() if eweights.ptp() else 1) * (ew_max - ew_min)
else:
    ew = []

k = 1.6 / np.sqrt(max(len(G.nodes), 1))
pos = nx.spring_layout(G, k=k, iterations=1000, seed=SEED, weight="weight")


fig, ax = plt.subplots(figsize=(12, 9))

nx.draw_networkx_edges(G, pos, ax=ax, width=ew, alpha=0.45, edge_color="gray")

nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_size=node_sizes,
    node_color=["#301934" if n=="Horror" else "#CC5500" for n in G.nodes],
    alpha=0.9, linewidths=0
)

labels = {n: n for n in G.nodes}
for n, (x, y) in pos.items():
    ax.text(
        x, y, labels[n],
        ha="center", va="center",
        fontsize=9,
        color="white" if n == "Horror" else "black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.65, edgecolor="none") if n != "Horror" else None
    )

ax.set_title("Popular Genres within Horror")
ax.axis("off")
fig.tight_layout()
plt.show()
#%%