import os
import shutil
from omegaconf import DictConfig
import hydra
import pytorch_lightning as pl
import torch

def reorganize_images(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_descriptive_text_ver2_adv_startGAN500_distill=0.5_adv=0.75"
    output_dir = "/home/xiyu/projects/AR-LDM/ckpts/img_AB_compare/diff_distill=0.25_adv=0.75"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(story_pth):
                    for img_file in os.listdir(story_pth):
                        if img_file.endswith("_original_eval.png") or img_file.endswith("_generated_eval.png"):
                            char_name = char_dir
                            story_id = img_file.split("_")[0]
                            img_id = img_file.split("_")[1]
                            new_filename = f"{char_name}-{story_id}-{img_id}.png"
                            src_path = os.path.join(story_pth, img_file)
                            dst_path = os.path.join(output_dir, new_filename)
                            shutil.copy(src_path, dst_path)
                            print(f"Copied {src_path} to {dst_path}")

    print("Image reorganization completed.")

@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    reorganize_images(args)

if __name__ == '__main__':
    main()