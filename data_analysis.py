import numpy as np, os
from tqdm import tqdm
from collections import defaultdict
from train import read_mask

import pandas as pd

def oversample(csv_path, output_size=None):
    df = pd.read_csv(csv_path, header=None, names=["filename", "class1", "class2"])

    total_class1 = df["class1"].sum()
    total_class2 = df["class2"].sum()

    if total_class1 > total_class2:
        smaller_class = "class2"
        larger_class = "class1"
    else:
        smaller_class = "class1"
        larger_class = "class2"

    oversampled_data = df.copy(deep=True)

    while oversampled_data[smaller_class].sum() < oversampled_data[larger_class].sum():
        sample = df[df[smaller_class] > 0].sample(n=1, replace=True)
        oversampled_data = pd.concat([oversampled_data, sample], ignore_index=True)

    if output_size:
        oversampled_data = oversampled_data.sample(n=output_size, replace=False, random_state=42)

    return oversampled_data


# create mask analysis

# mask_dir = 'data/train_gt/train_gt'
# train_files = os.listdir(mask_dir)
# train_files = sorted(train_files)
# with open('mask_analysis.csv', 'w') as fw:
#     for file in tqdm(train_files):
#         mask_image_path = f'{mask_dir}/{file}'
#         mask = read_mask(mask_image_path, None)
#         unique, count = np.unique(mask, return_counts=True)
#         pixel_dict = defaultdict(int, dict(zip(unique, count)))
#         fw.write(f'{file},{pixel_dict[1]},{pixel_dict[2]}\n')


csv_path = "mask_analysis.csv"
oversample_path = "oversampled_data.csv"

oversampled_df = oversample(csv_path)
oversampled_df.to_csv(oversample_path, index=False, header=False)

# test oversampling result
df_before = pd.read_csv(csv_path, header=None, names=["filename", "class1", "class2"])
print('Before:')
print(df_before["class1"].sum(), df_before["class2"].sum(), df_before["class1"].sum()/df_before["class2"].sum())

df_after = pd.read_csv(oversample_path, header=None, names=["filename", "class1", "class2"])
print('After:')
print(df_after["class1"].sum(), df_after["class2"].sum(), df_after["class1"].sum()/df_after["class2"].sum())