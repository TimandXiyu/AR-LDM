import os

src_dir = r"D:\AR-LDM\ckpts\po_user_std_start100\5_tutu"
target_dir = r"D:\AR-LDM\ckpts\po_user_std_start100\tutu_reorder"
# read all files in the target directory
files = os.listdir(src_dir)
# sort
files.sort()
# copy and rename to target directory
for i, file in enumerate(files):
    os.rename(os.path.join(src_dir, file), os.path.join(target_dir, f"{i}.png"))