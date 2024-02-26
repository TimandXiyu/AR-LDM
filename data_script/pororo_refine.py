import argparse
import os

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def main(args):
    descriptions = np.load(os.path.join(args.data_dir, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
    imgs_list = np.load(os.path.join(args.data_dir, 'img_cache4.npy'), encoding='latin1')

    followings_list = np.load(os.path.join(args.data_dir, 'following_cache4.npy'))
    train_ids, val_ids, test_ids = np.load(os.path.join(args.data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
    train_ids = np.sort(train_ids)
    val_ids = np.sort(val_ids)
    test_ids = np.sort(test_ids)
    all_ids = np.concatenate([train_ids, val_ids, test_ids])

    exclude_chars = ["harry"]

    marked_pth = list()
    print("exclude chars: ", exclude_chars)
    print("length of original descriptions is: ", len(descriptions))
    for id, desc in descriptions.items():
        for char in exclude_chars:
            if char in desc[0].lower():
                marked_pth.append(id + ".png")
    print("length of marked ids is: ", len(marked_pth))

    print("Starting creating HDF5 file")
    f = h5py.File(args.save_path, "w")
    for subset, ids in {'test': all_ids}.items():
        excluded_ids = list()
        for id in ids:
            cur_img_pth = [str(imgs_list[id])[2:-1]] + [str(followings_list[id][i])[2:-1] for i in range(4)]
            for pth in cur_img_pth:
                if pth in marked_pth:
                    excluded_ids.append(id)
        excluded_ids = list(set(excluded_ids))
        excluded_ids = sorted(excluded_ids)
        # removed marked ids
        print("length of original dataset is: ", len(ids))
        ids = [id for id in ids if id not in excluded_ids]
        length = len(ids)
        print("length of filtered dataset is: ", length)
        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            img_paths = [str(imgs_list[item])[2:-1]] + [str(followings_list[item][i])[2:-1] for i in range(4)]

            imgs = [Image.open(os.path.join(args.data_dir, img_path)).convert('RGB') for img_path in img_paths]
            for j, img in enumerate(imgs):
                img = np.array(img).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
            tgt_img_ids = [str(img_path).replace('.png', '') for img_path in img_paths]
            txt = [descriptions[tgt_img_id][0] for tgt_img_id in tgt_img_ids]
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones pororo file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='pororo data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    main(args)
