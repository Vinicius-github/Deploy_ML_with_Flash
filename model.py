#%%
#import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pickle

pd.set_option('display.max_columns', None)

# %%
#Load dataset
df_train = pd.read_csv('df_train_processed.csv')

# %%
df_train.drop(columns=['Unnamed: 0', 'model'], inplace = True)

# %%
df_train.info()

#%%
cars = df_train.drop("price", axis=1)
cars_price = df_train["price"].copy()

#%%
cars_num = cars.select_dtypes(include=['float64']).columns
cars_cat = cars.select_dtypes(include=['object']).columns

#%%
full_pipeline = ColumnTransformer([
        ("num", StandardScaler(), cars_num),
        ("cat", OneHotEncoder(), cars_cat)
    ])

#%%
model = Pipeline(steps=[
    ('pre_proc', full_pipeline),
    ('forest_regressor', RandomForestRegressor(random_state=42))
])

# %%
model.fit(cars, cars_price)
print('Trained model')

# %%
cars_predictions = model.predict(cars.iloc[:1000])
forest_mse = mean_squared_error(cars_price.iloc[:1000], 
                                cars_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

# %%
model = pickle.dump(model, open("model.pkl", "wb"))
