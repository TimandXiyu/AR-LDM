import torch
from torch.nn.functional import cosine_similarity
from omegaconf import DictConfig
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
import hydra

CUDA="cuda:0"

def get_dino_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_descriptive_text_ver2_200ep_forzenclipblip_refer"

    dino_net = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino_net = dino_net.to(CUDA)
    dino_net.eval()

    cosine_distances = {}

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            # sort char_dirs
            char_dirs = sorted(char_dirs, key=lambda x: int(x.split('_', 1)[0]))
            original_dino_features = []
            generated_dino_features = []
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(story_pth):
                    original_images = []
                    generated_images = []
                    img_paths = os.listdir(story_pth)
                    img_paths.remove('texts.json')
                    img_paths = sorted(img_paths, key=lambda x: int(x.split('_', 1)[0]))
                    for img_file in img_paths:
                        if img_file.endswith("_original_eval.png"):
                            original_images.append(os.path.join(story_pth, img_file))
                        elif img_file.endswith("_generated_eval.png"):
                            generated_images.append(os.path.join(story_pth, img_file))

                    if original_images and generated_images:
                        original_dino_features.extend(get_dino_features(dino_net, original_images))
                        generated_dino_features.extend(get_dino_features(dino_net, generated_images))

            if original_dino_features and generated_dino_features:
                original_dino_features = torch.stack(original_dino_features)
                generated_dino_features = torch.stack(generated_dino_features)
                cosine_distance = calculate_cosine_distance(original_dino_features, generated_dino_features)
                print(f"Average Cosine Distance for character {ckpt_dir}: {cosine_distance:.4f}")

                if ckpt_dir not in cosine_distances:
                    cosine_distances[ckpt_dir] = []
                cosine_distances[ckpt_dir].append(cosine_distance)

    # Save the cosine distances as a CSV file
    csv_file = "cos_scores/oneshot_training_descriptive_text_ver2_frozenclipblip_refer.csv"
    df = pd.DataFrame.from_dict(cosine_distances, orient='index', columns=["0"])
    df.index.name = 'Character'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    print(df)
    df.to_csv(csv_file)
    print(f"Cosine distances saved to {csv_file}")

def get_dino_features(dino_net, image_paths):
    dino_features = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            img_tensor = img_tensor.to(CUDA)
            features = dino_net(img_tensor)
        dino_features.append(features.squeeze())
    return dino_features

def calculate_cosine_distance(features1, features2):
    cosine_distances = 1 - cosine_similarity(features1, features2)
    return cosine_distances.mean().item()

@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    get_dino_metrics(args)

if __name__ == '__main__':
    main()