from deepdive.preproc.resize import resize_image_with_crop, resize_image_with_padding
import os

# Define paths
input_path = 'preprocessed_data/original_sizes/'
output_path = 'preprocessed_data/512x512/'
unused_path = 'preprocessed_data/unused_images/'

target_width = 512
target_height = 512

all_entries = os.listdir(input_path)

# Filter out subdirectories
subdirectories = [entry for entry in all_entries if os.path.isdir(os.path.join(input_path, entry))]

for subdirectory in subdirectories:
    resize_image_with_crop(input_path + subdirectory, output_path + 'cropped/'  + subdirectory, unused_path,
                 target_width, target_height)

    resize_image_with_padding(input_path + subdirectory, output_path + 'padded/'  + subdirectory, unused_path,
                 target_width, target_height)
