import os
from PIL import Image
import json
from base64 import b64decode

json_dir = './raw_data/CleanSea/'
output_dir = 'preprocessed_data/original_sizes/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [file for file in os.listdir(json_dir) if not file.startswith('.')]

for file_path in files:
    with open(json_dir + file_path) as file:
        json_data = json.load(file)

        filename = json_data.get('imagePath')
        img_data = json_data.get('imageData')

        # Check if imageData is not None
        if img_data is not None:
            img = b64decode(img_data)
            debris_type = json_data['shapes'][0]['label']

            if not os.path.exists(output_dir + debris_type):
                os.makedirs(output_dir + debris_type)

            with open(output_dir + debris_type + '/' + filename, 'wb') as open_file:
                open_file.write(img)
        else:
            print(f"Image data not found in {file_path}")
