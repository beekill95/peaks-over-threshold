# peaks-over-threshold

> [!WARNING]
> This is a quick implementation over the weekend,
> expect bugs, unpolished and untested code.

Implementation of the peaks-over-threshold (POT) algorithm to detect extreme values in time series data,
including Streaming POT (SPOT) and Streaming POT with drift (DSPOT).

The implementation follows the paper: _Siffer, Alban, et al.
"Anomaly detection in streams with extreme value theory."
Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
2017._

> [!NOTE]
> Unlike the original paper's implementation,
> this implementation reuses `L-BFGS-B` minimization results to speed-up the Grimshaw procedure.

> [!NOTE]
> If you encounter runtime warnings saying invalid values, it might be because of number underflow.
> Try casting input data to double-precision numbers.
> Additionally, if the time series values are too large,
> it might cause some issues with the upper and lower bounds of the Grimshaw procedure;
> in that case, try rescaling the time series to some smaller values depending on the domains.

__Sample Results__

_Please check the `examples` folder for more results._

![SPOT](./examples/spot.png)

![DSPOT](./examples/dspot.png)
