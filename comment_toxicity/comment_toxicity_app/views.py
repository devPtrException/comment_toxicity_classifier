from django.shortcuts import render
from joblib import load
import numpy as np
from . import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predictor(request):
    if request.method == "POST":

        # Extract form data

        # count  hate_speech_count  offensive_language_count  neither_count  class

        vectorizer = load(
            "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/vectorizer.joblib"
        )
        com_text = request.POST.get("com_text")
        com_text = vectorizer(str(com_text))

        # Load model and predict
        model = load(
            "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model.joblib"
        )
        # model = load("~/core/model/toxicity_model.joblib")

        # /home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model.joblib
        pred = model.predict(np.expand_dims(com_text, 0))
        # result_df = pd.DataFrame(result, columns=data.columns[1:])

        print("----------------------------------------------------Prediction:", (pred))
        context = {
            "count": float(pred[0][0]),  # Adjust based on your model's output
            "hate_speech_count": float(pred[0][1]),
            "offensive_language_count": float(pred[0][2]),
            "neither_count": float(pred[0][3]),
            "class": float(pred[0][4]),
        }

        return render(request, "result.html", {"result": context})
    return render(request, "result.html")


#
# # Create your views here.
#
#
# def predictor(request):
#     if request.method == "POST":
#         # Extract form data
#         com_text = request.POST.get("com_text")
#         if not com_text:
#             return render(request, "index.html", {"error": "No comment provided"})
#
#         try:
#             # Load model and tokenizer
#             model = joblib.load(
#                 "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model.joblib"
#             )
#             tokenizer = joblib.load(
#                 "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/tokenizer.joblib"
#             )
#
#             # Preprocess input
#             MAX_FEATURES = 20000  # From training
#             max_length = 100  # From training
#             # sequences = tokenizer.texts_to_sequences([com_text])
#             # padded = pad_sequences(sequences, maxlen=max_length)
#
#             com_text = vectorizer(com_text)
#             padded = pad_sequences(sequences, maxlen=max_length)
#
#             # Predict
#             pred = model.predict(com_text)[0]  # Shape (5,) for 5 labels
#             print("Prediction:", pred)
#
#             # Map predictions to context (assuming pred has 5 values)
#             context = {
#                 "count": float(pred[0]),  # Adjust based on your model's output
#                 "hate_speech_count": float(pred[1]),
#                 "offensive_language_count": float(pred[2]),
#                 "neither_count": float(pred[3]),
#                 "class": float(pred[4]),
#             }
#
#             return render(request, "result.html", {"result": context})
#         except Exception as e:
#             return render(request, "index.html", {"error": str(e)})
#
#     return render(request, "index.html")
#
#
# def predictori(request):
#     if request.method == "POST":
#
#         # Extract form data
#
#         # count  hate_speech_count  offensive_language_count  neither_count  class
#
#         com_text = str(request.POST.get("com_text"))
#         com_text = vectorizer(com_text)
#
#         # Load model and predict
#         model = load(
#             "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model.joblib"
#         )
#         # model = load("~/core/model/toxicity_model.joblib")
#
#         # /home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model.joblib
#         pred = model.predict(np.expand_dims(com_text, 0))
#
#         # result_df = pd.DataFrame(result, columns=data.columns[1:])
#
#         print("----------------------------------------------------Prediction:", (pred))
#         context = {
#             "count": count,
#             "hate_speech_count": hate_speech_count,
#             "offensive_language_count": offensive_language_count,
#             "neither_count": neither_count,
#             "class": classi,
#         }
#
#         return render(request, "result.html", {"result": context})
#     return render(request, "index.html")


# from django.shortcuts import render, redirect
# import joblib
# import numpy as np
# import pandas as pd
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# def predictor(request):
#     logger.info("Predictor view called")
#     if request.method == "POST":
#         com_text = request.POST.get("com_text")
#         logger.info(f"Comment text: {com_text}")
#         if not com_text:
#             logger.error("No comment provided")
#             return render(request, "index.html", {"error": "No comment provided"})
#
#         try:
#             # Load model and vectorizer
#             model = tf.keras.models.load_model(
#                 "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/toxicity_model"
#             )
#             vectorizer = joblib.load(
#                 "/home/mrmauler/DRIVE/projects/dl/comment_toxicity/core/model/vectorizer.joblib"
#             )
#             logger.info("Model and vectorizer loaded")
#
#             # Preprocess input (same as notebook)
#             input_txt = vectorizer([com_text])  # Vectorize the text
#             result = model.predict(np.expand_dims(input_txt, 0))  # Predict
#             logger.info(f"Prediction: {result}")
#
#             # Create DataFrame (same as notebook)
#             columns = [
#                 "count",
#                 "hate_speech_count",
#                 "offensive_language_count",
#                 "neither_count",
#                 "class",
#             ]
#             result_df = pd.DataFrame(result, columns=columns)
#             logger.info(f"Result DataFrame: {result_df}")
#
#             # Convert predictions to dict for rendering
#             prediction = result_df.iloc[0].to_dict()
#
#             # Store in session for redirect
#             request.session["prediction"] = prediction
#             return redirect("result")
#         except Exception as e:
#             logger.error(f"Error: {str(e)}")
#             return render(request, "index.html", {"error": str(e)})
#
#     logger.info("Rendering index.html for GET request")
#     return render(request, "index.html")
#
#
# def result(request):
#     prediction = request.session.get("prediction", {})
#     return render(request, "result.html", {"result": prediction})
