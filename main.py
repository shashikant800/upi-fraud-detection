import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# 🔴 Patch InputLayer (batch_shape fix)
from keras.layers import InputLayer
original_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    # Remove batch_shape but convert it to shape
    if "batch_shape" in kwargs:
        batch_shape = kwargs.pop("batch_shape")
        kwargs["input_shape"] = batch_shape[1:]   # (None, 10) → (10,)
    return original_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

# 🔴 Fix DTypePolicy issue
from keras.utils import custom_object_scope

with custom_object_scope({
    'DTypePolicy': tf.keras.mixed_precision.Policy
}):
    model = load_model('filesuse/project_model1.h5', compile=False)

print(model.summary())