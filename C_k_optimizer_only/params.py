from dataclasses import dataclass
import numpy as np

# --- DEFINE DROPLET DATACLASS ---
# Should contain all the information to define a droplet's deformation behavior
@dataclass
class DropletParams:
    target_deformation: float
    y_0_init: float
    A: float
    B_wo_C_k: float
    B: float
    C: float
    omega: float
    timespace: np.ndarray