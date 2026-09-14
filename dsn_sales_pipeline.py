"""
DSN Bootcamp Qualification Hackathon 2026 - ML Track
Predicting total_sales for a product at a given store (RMSE-scored regression)

Pipeline: EDA-informed cleaning -> feature engineering -> XGBoost (bagged) -> submission.csv
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ---------------------------------------------------------------------------
# 2. CLEANING / FEATURE ENGINEERING
#    Key data issues found during EDA:
#    - product_category has 48 raw values that are really 16 categories with
#      inconsistent casing (e.g. "Snack Foods" / "snack foods" / "SNACK FOODS")
#    - fat_content is meaningless for non-consumables (Household, Health and
#      Hygiene, Others) -> relabel as "Non_Edible"
#    - shelf_visibility has ~6-8% exact zeros, which is not physically
#      plausible for a stocked item -> treat as missing, impute by category mean
#    - product_weight_kg missing ~18% -> impute from other rows of the same
#      product_code (weight is a property of the product, near-constant across
#      stores with small measurement noise), falling back to category mean
#    - store_size missing for 3 of the 10 stores entirely (not random per-row) ->
#      kept as its own "Unknown" category; store_code itself already encodes
#      every store attribute uniquely (only 10 stores total)
# ---------------------------------------------------------------------------

NON_CONSUMABLE = {'Household', 'Health And Hygiene', 'Others'}
DRINKS = {'Soft Drinks', 'Hard Drinks'}


def normalize_category(s):
    return s.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True).str.title()


def engineer(df):
    df = df.copy()
    df['product_category'] = normalize_category(df['product_category'])

    def grp(c):
        if c in NON_CONSUMABLE:
            return 'Non_Consumable'
        if c in DRINKS:
            return 'Drinks'
        return 'Food'
    df['category_group'] = df['product_category'].apply(grp)

    df['fat_content_clean'] = df['fat_content'].where(
        df['category_group'] != 'Non_Consumable', 'Non_Edible'
    )

    df['shelf_visibility_raw'] = df['shelf_visibility']
    df.loc[df['shelf_visibility'] == 0, 'shelf_visibility'] = np.nan

    df['store_size'] = df['store_size'].fillna('Unknown')
    return df


def build_features(train, test):
    train_e = engineer(train)
    test_e = engineer(test)

    full = pd.concat(
        [train_e.drop(columns=['total_sales']), test_e], sort=False, ignore_index=True
    )

    # impute product_weight_kg: product-level mean -> category mean -> global mean
    full['product_weight_kg'] = full['product_weight_kg'].fillna(
        full.groupby('product_code')['product_weight_kg'].transform('mean')
    )
    full['product_weight_kg'] = full['product_weight_kg'].fillna(
        full.groupby('product_category')['product_weight_kg'].transform('mean')
    )
    full['product_weight_kg'] = full['product_weight_kg'].fillna(full['product_weight_kg'].mean())

    # impute shelf_visibility: category mean
    full['shelf_visibility'] = full['shelf_visibility'].fillna(
        full.groupby('product_category')['shelf_visibility'].transform('mean')
    )
    full['shelf_visibility'] = full['shelf_visibility'].fillna(full['shelf_visibility'].mean())

    # derived features
    full['price_per_kg'] = full['product_price'] / full['product_weight_kg']
    full['visibility_ratio'] = full['shelf_visibility'] / (
        full.groupby('product_category')['shelf_visibility'].transform('mean') + 1e-6
    )
    full['store_products_count'] = full.groupby('store_code')['store_code'].transform('count')
    full['product_stores_count'] = full.groupby('product_code')['product_code'].transform('count')

    n_train = len(train_e)
    train_out = full.iloc[:n_train].copy().reset_index(drop=True)
    test_out = full.iloc[n_train:].copy().reset_index(drop=True)
    train_out['total_sales'] = train_e['total_sales'].values
    return train_out, test_out


tr, te = build_features(train, test)

# Features actually used by the model (selected via 5-fold CV; the extra
# categoricals - fat_content, store_size, store_location_tier, category_group -
# added negligible/negative signal on top of store_code + product_category +
# store_format and were dropped)
CAT_COLS = ['product_category', 'store_code', 'store_format']
NUM_COLS = ['product_price', 'store_products_count', 'product_weight_kg',
            'price_per_kg', 'store_age_years', 'shelf_visibility', 'visibility_ratio']

full_oh = pd.get_dummies(
    pd.concat([tr[CAT_COLS + NUM_COLS], te[CAT_COLS + NUM_COLS]], sort=False, ignore_index=True),
    columns=CAT_COLS
)
X_train = full_oh.iloc[:len(tr)].reset_index(drop=True)
X_test = full_oh.iloc[len(tr):].reset_index(drop=True)
y_train = tr['total_sales'].values

# ---------------------------------------------------------------------------
# 3. CROSS-VALIDATION (sanity check before final fit)
#    5-fold CV RMSE with these settings: ~1080 (vs. a "predict the training
#    mean" baseline of ~1698, the std of total_sales)
# ---------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
)


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def cross_validate():
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        Xt, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        yt, yv = y_train[tr_idx], y_train[val_idx]
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE,
                                  early_stopping_rounds=100, eval_metric='rmse')
        model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        scores.append(rmse(yv, model.predict(Xv)))
    print(f"5-fold CV RMSE: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")
    return scores


cross_validate()

# ---------------------------------------------------------------------------
# 4. FINAL MODEL: bag 10 XGBoost models (different seeds / train-val splits)
#    and average predictions for a more stable leaderboard score.
# ---------------------------------------------------------------------------
preds = []
for seed in range(10):
    Xt, Xv, yt, yv = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
    model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed,
                              early_stopping_rounds=100, eval_metric='rmse')
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    preds.append(model.predict(X_test))

final_pred = np.clip(np.mean(preds, axis=0), 0, None)

# ---------------------------------------------------------------------------
# 5. SUBMISSION
# ---------------------------------------------------------------------------
submission = pd.DataFrame({'id': te['id'].values, 'total_sales': final_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows")
