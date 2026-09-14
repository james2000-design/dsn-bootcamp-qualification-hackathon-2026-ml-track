import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


def clean(df):
    df = df.copy()
    df['product_category'] = df['product_category'].str.strip().str.lower()
    df['fat_content'] = df['fat_content'].str.strip().str.lower()
    df['store_size'] = df['store_size'].str.strip()
    return df


train = clean(train)
test = clean(test)

full = pd.concat([train.drop(columns=['total_sales']), test], ignore_index=True, sort=False)

# shelf_visibility: 0 is implausible for a shelved product -> treat as missing
full.loc[full['shelf_visibility'] == 0, 'shelf_visibility'] = np.nan
full['shelf_visibility'] = full['shelf_visibility'].fillna(
    full.groupby('product_category')['shelf_visibility'].transform('mean')
)

# product_weight_kg: impute via product_code mean, then category mean, then global
full['product_weight_kg'] = full['product_weight_kg'].fillna(
    full.groupby('product_code')['product_weight_kg'].transform('mean')
)
full['product_weight_kg'] = full['product_weight_kg'].fillna(
    full.groupby('product_category')['product_weight_kg'].transform('mean')
)
full['product_weight_kg'] = full['product_weight_kg'].fillna(full['product_weight_kg'].mean())

# store_size: deterministic for 3 of 4 formats; else Unknown
format_to_size = {'Corner Shop': 'Small', 'Flagship Hypermarket': 'Medium', 'Superstore': 'Medium'}
mask = full['store_size'].isna()
full.loc[mask, 'store_size'] = full.loc[mask, 'store_format'].map(format_to_size)
full['store_size'] = full['store_size'].fillna('Unknown')

# feature engineering
full['price_per_kg'] = full['product_price'] / full['product_weight_kg']

cat_cols = ['fat_content', 'product_category', 'store_code',
            'store_location_tier', 'store_format']
num_cols = ['product_weight_kg', 'shelf_visibility', 'product_price',
            'store_age_years', 'price_per_kg']
feature_cols = cat_cols + num_cols

for c in cat_cols:
    full[c] = full[c].astype('category')

n_train = len(train)
X = full.iloc[:n_train][feature_cols].reset_index(drop=True)
X_test = full.iloc[n_train:][feature_cols].reset_index(drop=True)
y = train['total_sales'].values

params = dict(n_estimators=400, learning_rate=0.06, num_leaves=3,
              min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
              reg_alpha=0.0, reg_lambda=0.1, random_state=RANDOM_STATE, verbosity=-1)

# CV check
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, categorical_feature=cat_cols)
    pred = m.predict(X_val)
    scores.append(mean_squared_error(y_val, pred) ** 0.5)
print(f"5-fold CV RMSE: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

# Train final model on all training data
final_model = lgb.LGBMRegressor(**params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_preds = final_model.predict(X_test)
test_preds = np.clip(test_preds, 0, None)  # sales can't be negative

submission = pd.DataFrame({
    'id': test['id'].values,
    'total_sales': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Saved submission with", len(submission), "rows")
print(submission.head())
