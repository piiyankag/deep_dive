from PIL import Image, ImageOps
import os
from deepdive.preproc.utils import create_directory_if_not_exists

def resize_image_with_padding(input_path, output_path, unused_path, target_width, target_height):
    create_directory_if_not_exists(output_path)
    create_directory_if_not_exists(unused_path)

    for filename in os.listdir(input_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size

                # Discard if smaller than target size
                if width < target_width or height < target_height:
                    img.save(os.path.join(unused_path, filename))
                    continue

                # Resize with padding
                print('Resizing with padding: ' + filename)
                img = resize_with_padding(img, target_width, target_height)

                img.save(os.path.join(output_path, filename))

def resize_image_with_crop(input_path, output_path, unused_path, target_width, target_height):
    create_directory_if_not_exists(output_path)
    create_directory_if_not_exists(unused_path)

    for filename in os.listdir(input_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size

                # Discard if smaller than target size
                if width < target_width or height < target_height:
                    img.save(os.path.join(unused_path, filename))
                    continue

                # Center cropping or Adaptive scaling
                print('Center cropping or Adaptive scaling: ' + filename)
                img = center_crop_or_adaptive_scale(img, target_width, target_height)

                img.save(os.path.join(output_path, filename))

def resize_with_padding(img, target_width, target_height):
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio != target_ratio:
        # Padding
        img = ImageOps.pad(img, (target_width, target_height), color='black')
    else:
        img = img.resize((target_width, target_height))
    return img

def center_crop_or_adaptive_scale(img, target_width, target_height):
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Center cropping
        print('Center cropping selected')
        scale = target_height / img.height
        new_width = scale * img.width
        img = img.resize((int(new_width), target_height))
        left = (img.width - target_width) / 2
        img = img.crop((left, 0, left + target_width, target_height))
    else:
        # Adaptive scaling
        print('Adaptive scaling selected')
        scale = target_width / img.width
        new_height = scale * img.height
        img = img.resize((target_width, int(new_height)))
        top = (img.height - target_height) / 2
        img = img.crop((0, top, target_width, top + target_height))

    return img
