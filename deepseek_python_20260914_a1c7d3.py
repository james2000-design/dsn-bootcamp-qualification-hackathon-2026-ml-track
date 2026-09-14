"""
DSN Bootcamp Qualification Hackathon 2026 - ML Track
Improved pipeline: product target encoding + log target + 3-model blend.

Run:  python dsn_sales_pipeline_v2.py
Output: submission.csv
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ---------------------------------------------------------------------------
# 2. BASIC CLEANING
# ---------------------------------------------------------------------------
NON_CONSUMABLE = {'household', 'health and hygiene', 'others'}
DRINKS = {'soft drinks', 'hard drinks'}

def norm_cat(s):
    return s.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

def engineer(df):
    df = df.copy()
    df['product_category'] = norm_cat(df['product_category'])
    def grp(c):
        if c in NON_CONSUMABLE: return 'Non_Consumable'
        if c in DRINKS: return 'Drinks'
        return 'Food'
    df['category_group'] = df['product_category'].map(grp)
    df['store_size'] = df['store_size'].fillna('Unknown')
    df['fat_content'] = df['fat_content'].fillna('Unknown')
    return df

train = engineer(train)
test = engineer(test)

GLOBAL_MEAN = train['total_sales'].mean()

# ---------------------------------------------------------------------------
# 3. OUT-OF-FOLD TARGET ENCODING
# ---------------------------------------------------------------------------
def target_encode(train_df, test_df, keys, target='total_sales',
                  smooth=20, name=''):
    train_df = train_df.copy()
    test_df = test_df.copy()
    col = f'te_{name}'
    prior = GLOBAL_MEAN

    agg = train_df.groupby(keys)[target].agg(['mean', 'count'])
    sm = (agg['mean'] * agg['count'] + prior * smooth) / (agg['count'] + smooth)
    test_df[col] = test_df.set_index(keys).index.map(sm).fillna(prior)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_df[col] = np.nan
    for tr_idx, va_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        g = tr_fold.groupby(keys)[target].agg(['mean', 'count'])
        sm_f = (g['mean'] * g['count'] + prior * smooth) / (g['count'] + smooth)
        mapped = train_df.iloc[va_idx].set_index(keys).index.map(sm_f)
        train_df.iloc[va_idx, train_df.columns.get_loc(col)] = mapped.values
    train_df[col] = train_df[col].fillna(prior)
    return train_df, test_df

train_te, test_te = target_encode(train, test, ['product_code'], name='product')
train['te_product'] = train_te['te_product'].values
test['te_product']  = test_te['te_product'].values

train_te, test_te = target_encode(train, test,
                                  ['product_code', 'store_code'], name='prod_store')
train['te_prod_store'] = train_te['te_prod_store'].values
test['te_prod_store']  = test_te['te_prod_store'].values

train_te, test_te = target_encode(train, test,
                                  ['product_category', 'store_code'], name='cat_store')
train['te_cat_store'] = train_te['te_cat_store'].values
test['te_cat_store']  = test_te['te_cat_store'].values

train_te, test_te = target_encode(train, test, ['store_code'], name='store')
train['te_store'] = train_te['te_store'].values
test['te_store']  = test_te['te_store'].values

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
def build_features(df):
    df = df.copy()
    df['price_per_kg'] = df['product_price'] / df['product_weight_kg'].replace(0, np.nan)
    df['log_price'] = np.log1p(df['product_price'])
    df['log_weight'] = np.log1p(df['product_weight_kg'])
    df['is_zero_visibility'] = (df['shelf_visibility'] == 0).astype(int)
    df['shelf_visibility'] = df['shelf_visibility'].fillna(-1)
    df['store_product_count'] = df.groupby('store_code')['id'].transform('count')
    df['product_store_count'] = df.groupby('product_code')['id'].transform('count')
    return df

train = build_features(train)
test = build_features(test)

# ---------------------------------------------------------------------------
# 5. FEATURE MATRIX
# ---------------------------------------------------------------------------
LOW_CARD_CATS = ['product_category', 'category_group',
                 'store_size', 'store_location_tier', 'store_format',
                 'fat_content']

FEATURES_NUM = [
    'product_price', 'product_weight_kg', 'price_per_kg',
    'log_price', 'log_weight',
    'shelf_visibility', 'is_zero_visibility',
    'store_age_years',
    'store_product_count', 'product_store_count',
    'te_product', 'te_prod_store', 'te_cat_store', 'te_store',
]

X_train_full = train[FEATURES_NUM].copy().reset_index(drop=True)
X_test_full  = test[FEATURES_NUM].copy().reset_index(drop=True)

# One-hot encode categoricals (fit on combined train+test to align columns)
combined_cat = pd.concat([train[LOW_CARD_CATS], test[LOW_CARD_CATS]],
                         ignore_index=True)
dummies = pd.get_dummies(combined_cat, prefix=LOW_CARD_CATS, dummy_na=True)
n_tr = len(train)
X_train_full = pd.concat([X_train_full, dummies.iloc[:n_tr].reset_index(drop=True)], axis=1)
X_test_full  = pd.concat([X_test_full,  dummies.iloc[n_tr:].reset_index(drop=True)], axis=1)

y = train['total_sales'].values
y_log = np.log1p(y)

print(f"Train matrix: {X_train_full.shape}, Test matrix: {X_test_full.shape}")

# ---------------------------------------------------------------------------
# 6. MODELS with K-Fold OOF (training on log target)
# ---------------------------------------------------------------------------
def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))

def run_cv(model_fn, name):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_train_full))
    test_pred = np.zeros(len(X_test_full))
    for fold, (tr, va) in enumerate(kf.split(X_train_full)):
        m = model_fn()
        m.fit(X_train_full.iloc[tr], y_log[tr],
              eval_set=[(X_train_full.iloc[va], y_log[va])])
        oof[va] = m.predict(X_train_full.iloc[va])
        test_pred += m.predict(X_test_full) / N_FOLDS
    print(f"  {name}: raw RMSE = {rmse(y, np.expm1(oof)):.2f}")
    return oof, test_pred

print("Training XGBoost...")
xgb_oof, xgb_test = run_cv(lambda: xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=5, tree_method='hist',
    random_state=RANDOM_STATE, early_stopping_rounds=150,
    eval_metric='rmse'), 'XGB')

print("Training LightGBM...")
lgb_oof, lgb_test = run_cv(lambda: lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.02, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_STATE,
    verbose=-1), 'LGB')

print("Training CatBoost...")
cb_oof, cb_test = run_cv(lambda: CatBoostRegressor(
    iterations=3000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3.0, random_seed=RANDOM_STATE,
    verbose=0, early_stopping_rounds=150), 'CB')

# ---------------------------------------------------------------------------
# 7. BLEND (inverse-RMSE weights in log space)
# ---------------------------------------------------------------------------
w = np.array([1.0 / rmse(y_log, xgb_oof),
              1.0 / rmse(y_log, lgb_oof),
              1.0 / rmse(y_log, cb_oof)])
w = w / w.sum()
print(f"Blend weights: XGB={w[0]:.3f}  LGB={w[1]:.3f}  CB={w[2]:.3f}")

blend_oof  = w[0]*xgb_oof  + w[1]*lgb_oof  + w[2]*cb_oof
blend_test = w[0]*xgb_test + w[1]*lgb_test + w[2]*cb_test

print(f"FINAL OOF raw RMSE = {rmse(y, np.expm1(blend_oof)):.2f}")

# ---------------------------------------------------------------------------
# 8. SUBMISSION
# ---------------------------------------------------------------------------
final_pred = np.clip(np.expm1(blend_test), 0, None)

submission = pd.DataFrame({
    'id': test['id'].values,
    'total_sales': final_pred
})

# Sanity checks
assert len(submission) == len(test), "Row count mismatch!"
assert submission['id'].is_unique, "Duplicate IDs!"
assert submission['total_sales'].notna().all(), "NaN in predictions!"

submission.to_csv('submission.csv', index=False)
print(f"\n✓ Wrote submission.csv  ({len(submission)} rows)")
print(submission.head())
print("\nPrediction stats:")
print(submission['total_sales'].describe())