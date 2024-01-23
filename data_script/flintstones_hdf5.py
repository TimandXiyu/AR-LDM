import argparse
import json
import os
import pickle

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from PIL import Image


def main(args):
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    f = h5py.File(args.save_path, "w")
    for subset, ids in {'train': train_ids, 'val': val_ids, 'test': test_ids}.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4]
        length = len(ids)
        # check file exist in ids
        for id in ids:
            if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
                print("file not exist: {}".format(id))

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = [item] + followings[item]
            txt = list()
            for j, globalID in enumerate(globalIDs):
                img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
                img = np.concatenate(img, axis=0).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
                txt.append(descriptions[globalID])
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


def main_adaptation(args, N):
    """
    Choose samples that have particular words and compose them as the dataset for training
    """
    sentences = ["hoppy", "Hoppy"]
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    all_ids = train_ids + val_ids + test_ids
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    f = h5py.File(args.save_path, "w")
    ids = [i for i in all_ids if i in followings and len(followings[i]) == 4]

    # use the filtered id as dataset
    filtered_ids = [i for i in ids if any(sentence in descriptions[i] for sentence in sentences)]

    N = min(N, len(ids) - len(filtered_ids))  # Ensure N is not larger than available samples
    random_ids = random.sample([i for i in ids if i not in filtered_ids], N)

    ids = filtered_ids + random_ids
    length = len(ids)

    # check file exist in ids
    for id in ids:
        if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
            print("file not exist: {}".format(id))

    group = f.create_group('train')
    images = list()
    for i in range(5):
        images.append(
            group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
    text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
    for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
        globalIDs = [item] + followings[item]
        txt = list()

        for j, globalID in enumerate(globalIDs):
            img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
            img = np.concatenate(img, axis=0).astype(np.uint8)
            img = cv2.imencode('.png', img)[1].tobytes()
            img = np.frombuffer(img, np.uint8)
            images[j][i] = img
            txt.append(descriptions[globalID])

        annotation = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])

        if item in filtered_ids:
            # Display images and prompt for updated annotation
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))  # Adjust the figure size if needed
            for j in range(5):
                full_img = cv2.imdecode(images[j][i], cv2.IMREAD_COLOR)
                cropped_img = full_img[:128, :]  # Modify this line if a different cropping is needed
                axs[j].imshow(cropped_img)
                axs[j].axis('off')
            plt.show()

            # Request user input for updated annotation
            updated_annotation = input(
                f"Original annotation: {annotation}\nType the updated annotation (or press Enter to keep the original): ")
            if updated_annotation:
                annotation = updated_annotation

            text[i] = annotation
        else:
            text[i] = annotation

    f.close()


def main_refined_dataset(args):
    """
    Choose samples that does NOT have particular words and compose them as the dataset for training
    """
    removed_char = ["pebbles", "pearl slaghoople", "hoppy", "bamm bamm", "a policeman"]
    removed_char_by_id = [
        "s_05_e_22_shot_012349_012423",
        "s_05_e_22_shot_012424_012498",
        "s_05_e_22_shot_012914_012988",
        "s_05_e_22_shot_013506_013580",
        "s_05_e_22_shot_013881_013955",
        "s_05_e_22_shot_014520_014594",
        "s_05_e_22_shot_014652_014726",
        "s_05_e_22_shot_015112_015186",
        "s_05_e_22_shot_017501_017575",
        "s_05_e_22_shot_017829_017903",
        "s_05_e_22_shot_019272_019346",
        "s_05_e_22_shot_019437_019511",
        "s_05_e_22_shot_019737_019811",
        "s_05_e_22_shot_024101_024175",
        "s_05_e_22_shot_025630_025704",
        "s_05_e_22_shot_026950_027024",
        "s_05_e_22_shot_027025_027099",
        "s_05_e_22_shot_028265_028339",
        "s_05_e_22_shot_026216_026290"

    ]
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    raw_chars = dict()
    characters = dict()
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]
        raw_chars[sample["globalID"]] = sample["characters"]
    for k, v in raw_chars.items():
        characters[k] = [single_char["labelNPC"] for single_char in v]

    f = h5py.File(args.save_path, "w")
    # ids = [i for i in all_ids if i in followings and len(followings[i]) == 4]
    for subset, ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4]
        # check file exist in ids
        for id in ids:
            if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
                print("file not exist: {}".format(id))

        ids = [[i] + followings[i] for i in ids]
        filtered_id = []
        for i, story_id in enumerate(ids):
            for j, img_id in enumerate(story_id):
                if any(id.lower() in characters[img_id] for id in removed_char):
                    filtered_id.append(i)
        ids = [ids[i] for i in range(len(ids)) if i not in filtered_id]
        length = len(ids)

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = item
            txt = list()
            for j, globalID in enumerate(globalIDs):
                img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
                img = np.concatenate(img, axis=0).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
                txt.append(descriptions[globalID])

            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])

    f.close()


def get_img_from_descriptions(args):
    """
    Choose samples that does NOT have particular words and compose them as the dataset for training
    """
    sentences = ["The white mustached man in pink hat is stuck in the roof of the car.  He laughs about it."]
    target_ids = ["s_02_e_26_shot_012026_012100"]
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    all_ids = train_ids + val_ids + test_ids
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]
    # save descriptions as txt
    with open(os.path.join(args.data_dir, 'descriptions.txt'), 'w') as f:
        for key in descriptions:
            f.write(descriptions[key] + '\n')
    ids = [i for i in all_ids if i in followings and len(followings[i]) == 4]
    all_ids = [[i] + followings[i] for i in ids]
    filtered_id = []
    for i, story_id in enumerate(all_ids):
        for j, img_id in enumerate(story_id):
            if any(sentence in descriptions[img_id] for sentence in sentences):
                print(img_id, descriptions[img_id])
                filtered_id.append(i)

    filtered_id = [ids[i] for i in filtered_id]
    # get the images
    for id in filtered_id:
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))  # Create a plot with 1 row and 5 columns
        frame = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id)))
        frame = np.concatenate(frame, axis=0).astype(np.uint8)
        frame = frame[:128, :, :]
        axs[0].imshow(frame)  # Display the first frame
        axs[0].axis('off')

        for i, following_id in enumerate(followings[id]):
            frame = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(following_id)))
            frame = np.concatenate(frame, axis=0).astype(np.uint8)
            frame = frame[:128, :, :]
            axs[i + 1].imshow(frame)  # Display the following frames
            axs[i + 1].axis('off')

        plt.show()


def get_img_from_ids(args):
    # target_ids = ["s_05_e_01_shot_026411_026485",
    #               "s_05_e_10_shot_025883_025957",
    #               "s_05_e_20_shot_006919_006993",
    #               "s_05_e_06_shot_028741_028815",
    #               "s_05_e_01_shot_017675_017749",
    #               "s_05_e_16_shot_039116_039190"]
    # target_kw = "s_03_e_20_shot_029621_029695"
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    all_ids = train_ids + val_ids + test_ids
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]
    ids = [i for i in all_ids if i in followings and len(followings[i]) == 4]
    story_ids = {i: [i] + followings[i] for i in ids}

    # target_story = [story_ids[key] for key in story_ids if target_kw in key]

    # flatten the story_ids
    new_ids = []
    for story in story_ids:
        new_ids += story_ids[story]
    new_ids = list(set(new_ids))
    for ids in new_ids:
        # save each numpy array as jpg files named after the id
        frame = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(ids)))
        frame = np.concatenate(frame, axis=0).astype(np.uint8)
        frame = frame[:128, :, :]
        frame = Image.fromarray(frame)
        # save frame based on the seasons and episodes
        season = ids.split("_")[1]
        episode = ids.split("_")[3]
        # create folders for each episodes and save the frames
        if not os.path.exists(os.path.join(r"D:\AR-LDM\data\keyword_searching", season + "-" + episode)):
            os.mkdir(os.path.join(r"D:\AR-LDM\data\keyword_searching", season + "-" + episode))
        frame.save(os.path.join(r"D:\AR-LDM\data\keyword_searching", season + "-" + episode, str(ids) + ".jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    # get_img_from_descriptions(args)
    # main_refined_dataset(args)
    get_img_from_ids(args)