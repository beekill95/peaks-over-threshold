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
import seaborn as sns

from utils import fetch_ds

from peaks_over_threshold import SPOT, DSPOT

sns.set_theme()

# %% [markdown]
# # Air Quality

# %%
air_quality = fetch_ds(id=360)

# data (as pandas dataframes)
X = air_quality.data.features
print(X.info())
X.head()

# %%
sns.lineplot(X, x=X.index, y="PT08.S3(NOx)")

# %%
spot = SPOT(1e-4)
thresholds, alerts = spot.fit_predict(X["PT08.S3(NOx)"].to_numpy(), num_inits=1000)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(X, x=X.index, y="PT08.S3(NOx)")
ax.plot(X.index, thresholds)
ax.scatter(X.index[alerts], X["PT08.S3(NOx)"].loc[alerts], color="tomato", alpha=0.5)

# %%
spot = DSPOT(100, 1e-4)
thresholds, alerts = spot.fit_predict(X["PT08.S3(NOx)"].to_numpy(), num_inits=1000)

# %%
fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(X, x=X.index, y="PT08.S3(NOx)")
ax.plot(X.index, thresholds)
ax.scatter(X.index[alerts], X["PT08.S3(NOx)"].loc[alerts], color="tomato", alpha=0.5)
