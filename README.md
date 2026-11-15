# Mass Market Car Price Prediction (Australia)
 
Machine learning project to estimate the fair market price of mass-market vehicles from Australian listings, using age, mileage and physical specifications. The goal is an accurate and explainable model that helps buyers and sellers judge whether a car is fairly priced.
 
## Results
 
| Model | R² | Adj R² | MAE (AUD) | RMSE (AUD) |
|---|---|---|---|---|
| OLS (baseline) | 0.7807 | 0.7731 | 5,675 | 7,608 |
| OLS (log price) | 0.8072 | 0.8005 | 4,917 | 7,134 |
| Decision Tree | 0.8043 | 0.7976 | 4,902 | 7,187 |
| Random Forest | 0.8554 | 0.8504 | 4,283 | 6,177 |
| **XGBoost** | **0.8925** | **0.8888** | **3,686** | **5,326** |
 
All metrics are on the 20% held-out test set, on the raw price scale. XGBoost improves R² by 0.11 and cuts MAE by about 35% against the OLS baseline.
 
## Dataset
 
- File: `Australian Vehicle Prices.csv`
- About 16.7K listings after removing rows with no price, across 76 brands
- Features include Brand, Model, Year, UsedOrNew, Transmission, Engine, DriveType, FuelType, FuelConsumption, Kilometres, ColourExtInt, Location, CylindersinEngine, BodyType, Doors, Seats and Price (target)
## Workflow
 
1. **Cleaning**
   - Dropped `Title` and `Car/Suv`, and rows with a missing price
   - Replaced placeholders (`-`, `0`, `POA`, `- / -`) with nulls
   - Parsed engine capacity (CC), cylinders, fuel consumption, doors and seats into numeric fields
   - Extracted `State` from `Location`, and `ColourExt` from `ColourExtInt` (the interior colour was dropped, as about 46% of values were empty)
2. **Missing value imputation** (up to about 10.6% missing per column)
   - Within-model donor borrowing: `groupby(['Brand', 'Model'])` with forward and backward fill
   - Rule-based fills for the rest: electric vehicles set to 0 CC and 0 fuel consumption, body-type modes and medians, and condition-based mileage (demo, new, used)
   - `Unknown` category for State, Transmission and ColourExt
3. **Feature engineering**
   - `Age = 2023 - Year`
   - Dropped `Model` (781 categories) to avoid high-dimensional encoding, since Brand, BodyType, EngineCC, Cylinders and DriveType capture the market position
   - Grouped rare fuel types and colours into `Other` and `Rare`
4. **Outlier handling**
   - Removed listings under $2,500, then applied IQR filtering on price to focus on the mass market
   - Removed vehicles older than 22 years (99th percentile)
5. **EDA**: univariate distributions, price vs age, kilometres, engine size, cylinders and fuel consumption, and a correlation heatmap
6. **Preprocessing**: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features, 80/20 train-test split (`random_state=42`)
7. **Modelling and evaluation**
   - OLS baseline, with assumption tests: Shapiro-Wilk (normality), Breusch-Pagan (homoscedasticity), Durbin-Watson (autocorrelation) and VIF (multicollinearity)
   - The baseline failed linearity, normality and homoscedasticity. A log-transform on price fixed linearity, but the other failures remained, so tree-based models were evaluated
   - Decision Tree, Random Forest and XGBoost compared on R², Adjusted R², MAE and RMSE
8. **Interpretation**: XGBoost feature importance, summed from one-hot columns back to the original features
## Tech Stack
 
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, statsmodels, SciPy, XGBoost

 
