import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def main(file_path):
    data = np.load(file_path, allow_pickle=True)

    # Get all video arrays in the .npz file
    video_arrays = [data[file_name] for file_name in sorted(data.files)]

    # Define a function to display a batch of videos
    def display_batch(videos):
        # Determine the number of frames in the shortest video in the current batch
        min_frames = min(len(video) for video in videos)

        # Loop over each frame index
        for frame_idx in range(min_frames):
            # Update each window with the new frame
            for window_idx, video in enumerate(videos):
                # Convert RGB to BGR
                image_bgr = cv2.cvtColor(video[frame_idx], cv2.COLOR_RGB2BGR)
                # Display the image
                window_name = f'Video {window_idx}'
                cv2.imshow(window_name, image_bgr)

            # Wait for 33ms before displaying the next frame
            if cv2.waitKey(33) & 0xFF == ord('q'):  # Exit if 'q' is pressed
                return True  # Return True to indicate that we should exit

        return False  # Return False to indicate that we should continue

    # Create 5 windows for displaying videos
    for i in range(5):
        window_name = f'Video {i}'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Iterate over the videos in chunks of 5
    for i in range(0, len(video_arrays), 5):
        # Select the current batch of up to 5 videos
        current_videos = video_arrays[i:i+5]

        # Display the current batch of videos
        should_exit = display_batch(current_videos)
        if should_exit:
            break

    # When everything is done, release all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(file_path='../data/video_frames3.npz')
    # main_adaptation(args, N=4)