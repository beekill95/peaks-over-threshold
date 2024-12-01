# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import fetch_ds

from peaks_over_threshold import SPOT, DSPOT

sns.set_theme()

# %% [markdown]
# # Electric Power Consumption

# %%
ds = fetch_ds(id=235)

X = ds.data.features
print(X.info())
X.head()

# %%
X = X.replace({"?": pd.NA}).astype(
    {
        "Global_active_power": "Float64",
        "Global_reactive_power": "Float64",
        "Voltage": "Float64",
        "Global_intensity": "Float64",
        "Sub_metering_1": "Float64",
        "Sub_metering_2": "Float64",
        "Sub_metering_3": "Float64",
    }
)
print(X.info())
X.head()

# %%
# Drop all missing values.
X = X.dropna()
X = X.iloc[:100000]
X.info()

# %%
sns.lineplot(X, x=X.index, y="Global_active_power")

# %%
spot = SPOT(1e-4)
thresholds, alerts = spot.fit_predict(
    X["Global_active_power"].to_numpy(), num_inits=1000
)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(X, x=X.index, y="Global_active_power")
ax.plot(X.index, thresholds, label="Threshold")
ax.scatter(
    X.index[alerts],
    X["Global_active_power"].loc[alerts],
    color="tomato",
    alpha=0.5,
    label="Alerts",
)
ax.legend()
ax.set_title("SPOT")
fig.tight_layout()

# %%
spot = DSPOT(2 * 24 * 60, 1e-4)
thresholds, alerts = spot.fit_predict(
    X["Global_active_power"].to_numpy(), num_inits=1000
)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(X, x=X.index, y="Global_active_power")
ax.plot(X.index, thresholds)
ax.scatter(
    X.index[alerts],
    X["Global_active_power"].loc[alerts],
    color="tomato",
    alpha=0.5,
    label="Alerts",
)
ax.legend()
ax.set_title("DSPOT")
fig.tight_layout()
