# %%
import numpy as np
import ryenv
import lazyq

# %%
MY_ENV = ryenv.DiskEnv()

# %%
FLOOR_SIZE = 1
TARGET_UPDATE_INTERVAL = 2
N_EPOCHS = 10
NUM_BOOTSTRAPS = 6

# %%
V_FUNCTIONS = [
    lazyq.controllers.DnnRegressor2DPlus1D(
        FLOOR_SIZE,
        TARGET_UPDATE_INTERVAL,
        N_EPOCHS
    )
    for _ in range(NUM_BOOTSTRAPS)
]

# %%
LOCAL_MODEL_REGULARIZATION = 1e-5
DISCOUNT = 0.9
K_NEAREST = 20
BOUNDARIES = [-1, 1]
N_V_FUNCTION_UPDATE_ITERATIONS = 10
SCALEUP_FACTOR = 2.0**(1/3)

# %%
MY_CONTROLLER = lazyq.controllers.Controller(
    V_FUNCTIONS,
    LOCAL_MODEL_REGULARIZATION,
    DISCOUNT,
    K_NEAREST,
    BOUNDARIES,
    N_V_FUNCTION_UPDATE_ITERATIONS,
    MY_ENV,
    SCALEUP_FACTOR
)

# %%
