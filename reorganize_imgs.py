import os
import random
import shutil
import json
from PIL import Image
from omegaconf import DictConfig
import hydra
import torch
import pytorch_lightning as pl

def reorganize_images(args: DictConfig) -> None:
    data_dirs = [
        r"D:\AR-LDM\ckpts\pororo_textinv",
        r"D:\AR-LDM\ckpts\pororo_dreambooth",
        r"D:\AR-LDM\ckpts\pororo_customdiff",
        r"D:\AR-LDM\ckpts\pororo_desc_distill_0.25"
    ]
    output_dir = r"D:\AR-LDM\ckpts\ft_user_std_distill"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    char_dirs = [d for d in os.listdir(data_dirs[0]) if os.path.isdir(os.path.join(data_dirs[0], d))]

    for char_dir in char_dirs:
        story_dirs = [d for d in os.listdir(os.path.join(data_dirs[0], char_dir)) if
                      os.path.isdir(os.path.join(data_dirs[0], char_dir, d))]
        sampled_story_dirs = random.sample(story_dirs, min(3000, len(story_dirs)))

        char_output_dir = os.path.join(output_dir, char_dir)
        if not os.path.exists(char_output_dir):
            os.makedirs(char_output_dir)

        captions = []

        for idx, story_dir in enumerate(sampled_story_dirs, start=1):
            generated_images = []
            original_image = None
            caption = None
            image_idx = None

            for data_dir in data_dirs:
                story_path = os.path.join(data_dir, char_dir, story_dir)
                if not original_image:
                    original_images = [img for img in os.listdir(story_path) if img.endswith("_original_eval.png")]
                    if original_images:
                        original_image = os.path.join(story_path, original_images[0])

                gen_img = [img for img in os.listdir(story_path) if img.endswith("_generated_eval.png")]
                if gen_img:
                    if image_idx is None:
                        image_idx = random.randint(0, len(gen_img) - 1)
                    gen_img = gen_img[image_idx]  # Choose the randomly selected generated image
                    src_path = os.path.join(story_path, gen_img)
                    generated_images.append(src_path)

                    if not caption:
                        # Read the tests.json file and extract the corresponding caption
                        tests_json_path = os.path.join(story_path, "texts.json")
                        if os.path.exists(tests_json_path):
                            with open(tests_json_path, "r") as f:
                                tests_data = json.load(f)
                            img_index = int(gen_img.split("_")[0])  # Extract the image index from the filename
                            if 0 <= img_index < len(tests_data):
                                caption = tests_data[img_index]
                                captions.append(caption)
                            else:
                                raise ValueError(f"Image index {img_index} is out of bounds for {tests_json_path}")
                        else:
                            raise FileNotFoundError(f"Caption file {tests_json_path} not found for image {gen_img}")

            if original_image and generated_images:
                if not caption:
                    raise ValueError(f"No caption found for the selected image in story directory {story_dir}")

                # Concatenate the original image and generated images horizontally
                concat_image = Image.open(original_image)
                for img_path in generated_images:
                    img = Image.open(img_path)
                    concat_image = get_concat_h(concat_image, img)

                # Save the concatenated image
                concat_image_path = os.path.join(char_output_dir, f"{idx}.png")
                concat_image.save(concat_image_path)
                print(f"Saved concatenated image to {concat_image_path}")

        # Save the captions as a JSON file
        captions_path = os.path.join(char_output_dir, "captions.json")
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"Saved captions to {captions_path}")

    print("Image reorganization completed.")


def get_concat_h(im1, im2):
    # if first image is 128x128
    if im1.height != im2.height:
        im1 = im1.resize((im2.width, im2.height))
    dst = Image.new('RGB', (im1.width + im2.width, im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


@hydra.main(config_path="./config", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    reorganize_images(args)


if __name__ == '__main__':
    main()