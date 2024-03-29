import hydra
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from fid_utils import calculate_fid_given_features
from models.inception import InceptionV3
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import os
import json
import clip
from numpy import dot
from numpy.linalg import norm
import csv
import pandas as pd
from tqdm import tqdm

CUDA="cuda:0"


def get_metrics_singdir(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_original_text"

    evaluator = Evaluation(args)

    original_inception_features = []
    generated_inception_features = []

    for img_dir in tqdm(os.listdir(data_dir)):
        img_dir = os.path.join(data_dir, img_dir)
        for img in os.listdir(img_dir):
            if img.endswith("_original.png"):
                original_inception_features.extend(evaluator.get_inception_feature([os.path.join(img_dir, img)]))
            elif img.endswith("_generated.png"):
                generated_inception_features.extend(evaluator.get_inception_feature([os.path.join(img_dir, img)]))

    if original_inception_features and generated_inception_features:
        original_inception_features = np.vstack(original_inception_features)
        generated_inception_features = np.vstack(generated_inception_features)
        fid = calculate_fid_given_features(original_inception_features, generated_inception_features)
        print(f"FID for the folder: {fid}")
    else:
        print("No images found in the folder.")


# Rest of the code remains the same


def get_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_descriptive_text_ver2_200ep_forzenclipblip_SAMPLINGrefer"

    evaluator = Evaluation(args)
    fid_scores = {}

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            original_inception_features = []
            generated_inception_features = []
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(story_pth):
                    original_images = []
                    generated_images = []
                    for img_file in os.listdir(story_pth):
                        if img_file.endswith("_original_eval.png"):
                            original_images.append(os.path.join(story_pth, img_file))
                        elif img_file.endswith("_generated_eval.png"):
                            generated_images.append(os.path.join(story_pth, img_file))

                    if original_images and generated_images:
                        original_inception_features.extend(evaluator.get_inception_feature(original_images))
                        generated_inception_features.extend(evaluator.get_inception_feature(generated_images))

            if original_inception_features and generated_inception_features:
                original_inception_features = np.vstack(original_inception_features)
                generated_inception_features = np.vstack(generated_inception_features)
                fid = calculate_fid_given_features(original_inception_features, generated_inception_features)
                print(f"FID for character {ckpt_dir}: {fid}")

                if ckpt_dir not in fid_scores:
                    fid_scores[ckpt_dir] = []
                fid_scores[ckpt_dir].append(fid)

    # Save the FID scores as a CSV file
    csv_file = "fid_scores/oneshot_training_descriptive_text_ver2_frozenclipblip.csv"
    df = pd.DataFrame.from_dict(fid_scores, orient='index', columns=["0"])
    df.index.name = 'Character'

    print(df)
    df.to_csv(csv_file)
    print(f"FID scores saved to {csv_file}")

def get_unifed_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_descriptive_text_ver2_200ep_forzenclipblip"

    evaluator = Evaluation(args)

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))
    original_inception_features = []
    generated_inception_features = []
    for ckpt_dir in tqdm(ckpt_dirs):
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(story_pth):
                    original_images = []
                    generated_images = []
                    for img_file in os.listdir(story_pth):
                        if img_file.endswith("_original_eval.png") or img_file.endswith("_original.png"):
                            original_images.append(os.path.join(story_pth, img_file))
                        elif img_file.endswith("_generated_eval.png") or img_file.endswith("_generated.png"):
                            generated_images.append(os.path.join(story_pth, img_file))

                    if original_images and generated_images:
                        original_inception_features.extend(evaluator.get_inception_feature(original_images))
                        generated_inception_features.extend(evaluator.get_inception_feature(generated_images))

    if original_inception_features and generated_inception_features:
        original_inception_features = np.vstack(original_inception_features)
        generated_inception_features = np.vstack(generated_inception_features)
        fid = calculate_fid_given_features(original_inception_features, generated_inception_features)
        print(f"FID for all character: {fid}")

class Evaluation(object):
    """
    Hosting fid model for evaluation
    """
    def __init__(self, args):
        self.device = CUDA
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid_model = InceptionV3([block_idx]).to(self.device)
        self.fid_model.eval()
        self.fid_augment = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def get_inception_feature(self, images):
        images = [Image.open(img) for img in images]
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.fid_model(images)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048).cpu().numpy()


@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    # get_metrics(args)
    get_unifed_metrics(args)
    # get_metrics_singdir(args)

if __name__ == '__main__':
    main()
