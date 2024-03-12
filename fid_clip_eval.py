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


def get_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/output_images_3_adapted_conti_40stop_distill/"

    evaluator = Evaluation(args)
    fid_scores = []

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for k, ckpt_dir in enumerate(ckpt_dirs):
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            ckpt_fid_scores = []

            characters = [d.split('_', 1)[1] for d in ckpt_dirs[:ckpt_dirs.index(ckpt_dir) + 1]]
            char_dirs = sorted(os.listdir(ckpt_path), key=lambda x: characters.index(x))

            for char_dir in char_dirs:
                char_path = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(char_path):
                    original_inception_features = []
                    generated_inception_features = []

                    for story_dir in os.listdir(char_path):
                        story_path = os.path.join(char_path, story_dir)
                        if os.path.isdir(story_path):
                            original_images = []
                            generated_images = []

                            for img_file in os.listdir(story_path):
                                if img_file.endswith("_original_eval.png"):
                                    original_images.append(os.path.join(story_path, img_file))
                                elif img_file.endswith("_generated_eval.png"):
                                    generated_images.append(os.path.join(story_path, img_file))

                            if original_images and generated_images:
                                original_inception_features.extend(evaluator.get_inception_feature(original_images))
                                generated_inception_features.extend(evaluator.get_inception_feature(generated_images))

                    if original_inception_features and generated_inception_features:
                        original_inception_features = np.vstack(original_inception_features)
                        generated_inception_features = np.vstack(generated_inception_features)
                        fid = calculate_fid_given_features(original_inception_features, generated_inception_features)
                        print(f"FID for character {char_dir} in {ckpt_dir}: {fid}")
                        ckpt_fid_scores.append(fid)

            fid_scores.append(ckpt_fid_scores)

    # Save the FID scores as a CSV file
    csv_file = "fid_scores.csv"
    data = {}
    for i, ckpt_fid_scores in enumerate(fid_scores):
        data[f'Checkpoint {i}'] = ckpt_fid_scores

    characters = [d.split('_', 1)[1] for d in ckpt_dirs]
    max_length = max(len(values) for values in data.values())

    for key in data:
        data[key].extend([float('nan')] * (max_length - len(data[key])))

    row_names = [i for i in characters]
    df = pd.DataFrame.from_dict(data, orient='index', columns=row_names)

    print(df.T)
    df.T.to_csv(csv_file)
    print(f"FID scores saved to {csv_file}")

class Evaluation(object):
    """
    Hosting fid model for evaluation
    """
    def __init__(self, args):
        self.device = "cuda:0"
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


@hydra.main(config_path=".", config_name="config-eval")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    get_metrics(args)


if __name__ == '__main__':
    main()
