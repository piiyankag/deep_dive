from deepdive.params import *
from deepdive.ml_logic.data import download_data, load_data
from deepdive.ml_logic.model import build_model
from deepdive.ml_logic.preprocessor import preprocess_dataset, load_and_preprocess_image
from deepdive.ml_logic.registry import save_results, save_model, load_model

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from tensorflow.keras.callbacks import EarlyStopping


def preprocess() -> None:
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)
    download_data()

    train_ds, val_ds, class_names = load_data()

    train_ds = train_ds.map(preprocess_dataset)
    val_ds = val_ds.map(preprocess_dataset)

    print("✅ preprocess() done \n")

    return train_ds, val_ds, class_names


def train() -> float:

    """
    - Download images (or from cache if it exists)
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return val_mae as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: train" + Style.RESET_ALL)
    print(Fore.BLUE + "\nLoading preprocessed validation data..." + Style.RESET_ALL)

    train_ds, val_ds, class_names = preprocess()

    model = build_model(train_ds, class_names)

    es = EarlyStopping(monitor = 'val_accuracy',
                   mode = 'max',
                   patience = 5,
                   verbose = 1,
                   restore_best_weights = True)

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=50,
                        callbacks=[es])

    val_f1_score = np.max(history.history['val_f1_score'])

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_f1_score


def pred(url: str = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSc4w6Ey0a05Et7my42NDnAn9CTvbrFx8CSmA&usqp=CAU") -> str:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    model = load_model()
    assert model is not None

    X_processed = load_and_preprocess_image(url)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    train()
    pred()