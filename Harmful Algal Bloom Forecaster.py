# imports
import marimo as mo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts.metrics import *
from darts import TimeSeries
from darts.utils.utils import generate_index
from darts.models import RandomForestModel, XGBModel, LightGBMModel, CatBoostModel

# Satellite Data Preprocessing

# ====================
# Use panda dataframe (darts work with pandas)
# ====================

#from google.colab import drive
#drive.mount('/content/drive', force_remount = True)
#train = '/content/drive/MyDrive/HABs/DailyDelhiClimateTrain.csv'                 # Train Data
#train = pd.read_csv(train)

# Individual Graph
train['date'] = pd.to_datetime(train['date'])
# Removed this line: train.set_index('date', inplace=True)

for col in ['meantemp', 'humidity', 'wind_speed']:
    plt.figure(figsize=(23.5, 7))
    train[col].plot()
    plt.title(col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.grid(True)
    plt.show()

# Combined Graph
plt.figure(figsize=(23.5, 7))                                                    # Added figure size
plt.plot(train.index, train['meantemp'], color='r', label ='Mean Temperature')
plt.plot(train.index, train['humidity'], color='g', label ='Humidity')
plt.plot(train.index, train['wind_speed'], color='b', label = 'Wind Speed')
plt.xlabel("Date")
plt.ylabel("Magnitude")
plt.title("Mean Temperature, Humidity, and Wind Speed over Time")
plt.legend()
plt.grid(True)
plt.show()



# Test train split
df = pd.DataFrame(train)
df = df.drop('meanpressure', axis=1)
series = TimeSeries.from_dataframe(df, time_col="date")

train, val = series.split_before(pd.Timestamp("20160701"))

plt.figure(figsize=(23.5, 7))
train.plot(label="training")
val.plot(label="validation");




# Random Forest

# target = series['p (humidity)'][:100]
# past_cov = series['rain (mm)'][:100]
# future_cov = series['T (degC)'][:106]
model = RandomForestModel(
    lags=12,
    # lags_past_covariates=12,
    # lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
    n_estimators=200,
    criterion="absolute_error",
)

model.fit(train)
pred = model.predict(184)
pred.values()


# Random Forets print
plt.figure(figsize=(23.5, 7))
series.plot()

pred.plot(label="forecast");
train.plot(label="training")
#pred.plot(label="naive forecast (K=12)");

# Added val as the first argument to mae
print(f"Mean absolute error for Random Forest model: {mae(val, pred):.2f}%.") # Changed label to MAE for clarity


# LightGBM
#target =  series['rain (mm)'][:105].map(lambda x: np.where(x > 0, 1, 0))
#past_cov = series['T (degC)'][:105]
#future_cov = series['p (mbar)'][:111]

model = LightGBMModel(
    lags=12,
    #lags_past_covariates=12,
    #lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
    verbose=-1,
    criterion = "absolute_error",
)
model.fit(train)
pred = model.predict(187)
pred.values()


# Random Forets print
plt.figure(figsize=(23.5, 7))
series.plot()

pred.plot(label="forecast");
train.plot(label="training")
#pred.plot(label="naive forecast (K=12)");

# Added val as the first argument to mae
print(f"Mean absolute error for Random Forest model: {mae(val, pred):.2f}%.") # Changed label to MAE for clarity



# XGBoost

model = XGBModel(
    lags=12,
    #lags_past_covariates=12,
    #lags_future_covariates=[0,1,2,3,4,5],
    output_chunk_length=6,
    criterion = "absolute_error",
)
model.fit(train)
pred = model.predict(187)
pred.values()


plt.figure(figsize=(23.5, 7))
series.plot()

pred.plot(label="forecast");
train.plot(label="training")
#pred.plot(label="naive forecast (K=12)");

# Added val as the first argument to mae
print(f"Mean absolute error for Random Forest model: {mae(val, pred):.2f}%.") # Changed label to MAE for clarity

from darts.models import NHiTSModel

#target = series['p (mbar)'][:100]
#past_cov = series['rain (mm)'][:100]

model = NHiTSModel(
    input_chunk_length=6,
    output_chunk_length=6,
    num_blocks=2,
    n_epochs=500,
)
model.fit(train)
pred = model.predict(187)
print(pred.values())



plt.figure(figsize=(23.5, 7))
series.plot()

pred.plot(label="forecast");
train.plot(label="training")
#pred.plot(label="naive forecast (K=12)");


from darts.models import NBEATSModel

#target = series['p (mbar)'][:100]
#past_cov = series['rain (mm)'][:100]

model = NHiTSModel(
    input_chunk_length=6,
    output_chunk_length=6,
    num_blocks=2,
    n_epochs=500,
)
model.fit(train)
pred = model.predict(187)
print(pred.values())

#NBEATS

from darts.metrics import r2_score
from darts.models import NBEATSModel
from darts.utils.callbacks import TFMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint # Import ModelCheckpoint

def generate_torch_kwargs():
    # run torch models on CPU, and disable progress bars for all model stages except training.
    return {
        "pl_trainer_kwargs": {
            "accelerator": "cpu",
            "callbacks": [
                TFMProgressBar(enable_train_bar_only=True),
                ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1) # Monitor train_loss
            ],
        }
    }

model_name = "nbeats_interpretable_run"
model_nbeats = NBEATSModel(
    input_chunk_length=20, # Reduced input chunk length
    output_chunk_length=5, # Reduced output chunk length
    generic_architecture=False,
    num_blocks=3,
    num_layers=4,
    layer_widths=512,
    n_epochs=100,
    nr_epochs_val_period=1,
    batch_size=128,
    random_state=42,
    model_name=model_name,
    save_checkpoints=True,
    force_reset=True,
    **generate_torch_kwargs(),
)

# Explicitly set covariates to None
model_nbeats.fit(train, val_series=val, past_covariates=None, future_covariates=None)
model_nbeats = NBEATSModel.load_from_checkpoint(model_name=model_name, best=True)

pred_nbeats = model_nbeats.predict(len(val)) # Predict on the length of the validation set

plt.figure(figsize=(23.5, 7))
series.plot()
pred_nbeats.plot(label="NBEATS forecast");
train.plot(label="training")

print(f"Mean absolute error for NBEATS model: {mae(val, pred_nbeats):.2f}%.")


# Model Evaluation and Validation

