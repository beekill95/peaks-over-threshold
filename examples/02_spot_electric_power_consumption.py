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

from peaks_over_threshold import SPOT

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
        "Global_active_power": "Float32",
        "Global_reactive_power": "Float32",
        "Voltage": "Float32",
        "Global_intensity": "Float32",
        "Sub_metering_1": "Float32",
        "Sub_metering_2": "Float32",
        "Sub_metering_3": "Float32",
    }
)
print(X.info())
X.head()

# %%
# Drop all missing values.
X = X.dropna()
X.info()

# %%
sns.lineplot(X, x=X.index, y="Global_active_power")

# %%
spot = SPOT()
spot.fit_predict(X["Global_active_power"].to_numpy(), num_inits=1000)
