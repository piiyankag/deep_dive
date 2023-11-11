from deepdive.preproc.extractor import extract_images

json_dir = './raw_data/CleanSea/'
output_dir = 'preprocessed_data/original_sizes/'

extract_images(json_dir, output_dir)
