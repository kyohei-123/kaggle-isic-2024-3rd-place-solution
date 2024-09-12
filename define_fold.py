# %%
import pandas as pd
from sklearn.model_selection import GroupKFold

# %%

df_train = pd.read_csv("./input/isic-2024-challenge/train-metadata.csv")

gkf = GroupKFold(n_splits=5)

df_train["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df_train, df_train["target"], groups=df_train["patient_id"])):
    df_train.loc[val_idx, "fold"] = idx

# Add summary
fold_summary = df_train.groupby("fold")["patient_id"].nunique().to_dict()
total_patients = df_train["patient_id"].nunique()

print(f"Fold Summary (patients per fold):")
for fold, count in fold_summary.items():
    if fold != -1:  # Exclude the initialization value
        print(f"Fold {fold}: {count} patients")
print(f"Total patients: {total_patients}")

df_train[["isic_id", "fold"]].head(10)
df_train[["isic_id", "fold"]].tail(10)

df_train[["isic_id", "fold"]].to_csv("./df_fold.csv", index=False)

# %%
