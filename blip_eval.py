import torch
from torch.nn.functional import cosine_similarity
from omegaconf import DictConfig
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import pytorch_lightning as pl
import hydra
import json
from transformers import BlipProcessor, BlipModel

CUDA="cuda:0"

def get_blip_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/generated_oneshot_9unseen_descriptive_text_ver2_adv_startGAN500_distill=0.5_adv=0.75"

    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = blip_model.to(CUDA)
    blip_model.eval()

    similarity_scores = {}

    # Sort the checkpoint folders based on their numerical prefix
    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            # sort char_dirs
            char_dirs = sorted(char_dirs, key=lambda x: int(x.split('_', 1)[0]))
            char_similarity_scores = []
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                if os.path.isdir(story_pth):
                    generated_images = []
                    img_paths = os.listdir(story_pth)
                    img_paths = [path for path in img_paths if path != 'texts.json']  # Exclude 'texts.json' from img_paths
                    img_paths = sorted(img_paths, key=lambda x: int(x.split('_', 1)[0]))
                    for img_file in img_paths:
                        if img_file.endswith("_generated_eval.png"):
                            generated_images.append(os.path.join(story_pth, img_file))

                    if generated_images:
                        # Load the texts from texts.json
                        with open(os.path.join(story_pth, "texts.json"), "r") as f:
                            texts = json.load(f)

                        # Calculate the text-image similarity for each image in the story
                        similarities = calculate_blip_similarity(blip_processor, blip_model, generated_images, texts)
                        avg_similarity = sum(similarities) / len(similarities)
                        char_similarity_scores.append(avg_similarity)

            if char_similarity_scores:
                avg_char_similarity = sum(char_similarity_scores) / len(char_similarity_scores)
                similarity_scores[ckpt_dir] = avg_char_similarity
                print(f"Average Text-Image Similarity for character {ckpt_dir}: {avg_char_similarity:.4f}")

    # Save the similarity scores as a CSV file
    csv_file = "similarity_scores/test.csv"
    df = pd.DataFrame.from_dict(similarity_scores, orient='index', columns=['Average Similarity'])
    df.index.name = 'Character'
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    print(df)
    df.to_csv(csv_file)
    print(f"Similarity scores saved to {csv_file}")

def calculate_blip_similarity(blip_processor, blip_model, image_paths, texts):
    similarities = []
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        text = texts[i]  # Get the corresponding text for the current image
        inputs = blip_processor(img, text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(CUDA) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = blip_model(**inputs)
        similarity = outputs.logits_per_image.item()
        similarities.append(similarity)
    return similarities

@hydra.main(config_path=".", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    get_blip_metrics(args)

if __name__ == '__main__':
    main()