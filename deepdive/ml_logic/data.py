from tensorflow.keras.utils import image_dataset_from_directory, to_categorical
import requests
import os
import zipfile

def clean_downloads():
    data_dir = os.environ.get("DATA_PATH")
    if os.path.exists(data_dir):
        print('Cleaning up data directory')
        os.system('rm -rf {}'.format(data_dir))

def download_data():
    if(os.path.exists(os.environ.get("DATA_PATH"))):
        print('Data already downloaded')
        return

    url = os.environ.get('DATA_URL')
    data_dir = os.environ.get("DATA_EXTRACT_PATH")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data_file = data_dir + "/photos.zip"
    if not os.path.exists(data_file):
        r = requests.get(url)
        with open(data_file, 'wb') as f:
            f.write(r.content)

    with zipfile.ZipFile(data_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)

    print('Data downloaded and extracted to {}'.format(data_dir))

def get_class_names():
    data_dir = os.environ.get("DATA_PATH")
    return os.listdir(data_dir)

def load_data():
    data_dir = os.environ.get("DATA_PATH")

    batch_size = 32
    img_height = 256
    img_width = 256

    # prepare train set
    train_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    # prepare val set
    val_ds = image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    for image_batch, labels_batch in train_ds:
        print('train ds shape')
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    return train_ds, val_ds, class_names
