from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from deepdive.ml_logic.preprocessor import load_and_preprocess_image, preprocess_image
from deepdive.ml_logic.registry import load_model
from deepdive.ml_logic.data import get_class_names
import numpy as np

""""
Y ôLô Y
|  º  |
"""

app = FastAPI()
app.state.model = load_model()
app.state.class_names = get_class_names()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Make a single image prediction.
    """
    contents = await file.read()

    X_processed = preprocess_image(contents)
    y_pred = app.state.model.predict(X_processed)

    # Process predictions
    predicted_class = np.argmax(y_pred, axis=1)

    top_3_indices = np.argsort(y_pred[0])[-3:][::-1]
    top_3_class_names = [app.state.class_names[idx] for idx in top_3_indices]


    # Get the probabilities of the top 3 predictions
    top_3_probs = y_pred[0][top_3_indices]

    print("\n✅ Top 3 Predictions:")
    for idx, prob in zip(top_3_indices, top_3_probs):
        print(f"Class {idx}: Probability {prob}")

    print("\nPrediction details: ", y_pred, y_pred.shape, "\n")

    #return top_3_indices, top_3_probs

    return {
        'category': app.state.class_names[int(predicted_class)],
        'meta': zip(top_3_class_names, top_3_probs.tolist())
    }


@app.get("/")
def root():
    return {
        "greeting": "Hello"
    }
