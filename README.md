# Vomitoxin-Prediction
The dataset used in this project contains hyperspectral reflectance values along with a target variable vomitoxin_ppb (DON concentration in parts per billion).
Total Samples: ~2000 (approximate)
Total Features: 449
448 Spectral Features → Represent reflectance values at different wavelengths.
1 Target Variable (vomitoxin_ppb) → Indicates DON concentration in grains.

1️⃣ Preprocessing Steps & Rationale
The dataset consists of hyperspectral reflectance values across multiple wavelengths, with the target variable being vomitoxin concentration (DON in ppb). The following preprocessing steps were applied:

✅ Handling Missing Data:

Checked for null values and found none, ensuring data integrity.
✅ Feature Scaling:

Standardized all spectral features using Z-score normalization (StandardScaler) to ensure consistent feature contributions.
✅ Anomaly Detection & Outlier Removal:

Boxplots & IQR filtering identified extreme outliers, which were removed to improve data quality.
Log transformation was applied to vomitoxin_ppb to reduce skewness.
✅ Exploratory Data Analysis (EDA):

Histograms & boxplots were used to examine distributions.
Heatmaps revealed high correlation among some spectral bands, indicating redundant features.
✅ Sensor Drift & Spectral Index Computation:

Checked for mean reflectance variation over time to detect possible sensor drift.
Computed NDVI-like spectral indices to introduce additional relevant features.
2️⃣ Insights from Dimensionality Reduction
✅ Principal Component Analysis (PCA):

Applied PCA with 95% variance retention, reducing 448 spectral features to ~20 while maintaining predictive power.
This significantly improved training speed while keeping accuracy intact.
Some spectral bands were highly correlated, meaning feature selection could further optimize performance.
3️⃣ Model Selection, Training & Evaluation
📌 Model Chosen:
✅ Feedforward Neural Network (FNN) with (256 → 128 → 1 architecture).
✅ Compared with Random Forest & XGBoost, but FNN performed best.
✅ Adam Optimizer (lr=0.001), MSE Loss Function for robust learning.

**📌 Hyperparameter Tuning:**
✅ Grid Search was used to optimize:

Learning rate: 0.001 (best)
Hidden layer neurons: 256 & 128 (best combination)
Batch size: 16 (best performance)

**📌 Evaluation Metrics:**

**Metric	Score**
MAE (Mean Absolute Error)	12.45
RMSE (Root Mean Squared Error)	15.32
R² Score	0.82
R² = 0.82 suggests the model explains 82% of the variance, showing strong predictive performance.
Residual analysis showed no systematic bias.
