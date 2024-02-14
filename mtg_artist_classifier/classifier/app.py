import json
import os

import mlflow
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from prepare_data_constants import IMAGE_TRANSFORMS

app = Flask(__name__)


def get_model():
    device = torch.device("cpu")
    model = mlflow.pytorch.load_model("./model/", map_location=device)
    return model


def get_artist_dict() -> dict[str, str]:
    with open(os.path.join("model", "code", "artist_mapping.json"), "r") as fp:
        artist_mapping = json.load(fp)
    return artist_mapping


@app.route("/image", methods=["GET", "POST"])
def predict_unnormalized_image_as_list():
    if request.method == "POST":
        file = request.files["image"]
        image = Image.open(file)
        # return str(type(image))
        artist_mapping = get_artist_dict()
        model = get_model()
        tensor_input = IMAGE_TRANSFORMS(image)
        tensor_input = tensor_input.unsqueeze(0)
        logits = model(tensor_input)
        with torch.no_grad():
            prediction = int(torch.argmax(logits))
        str_prediction = str(prediction)
        return jsonify(
            {
                "prediction": artist_mapping[str_prediction],
            }
        )
    return """
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image">
      <input type="submit">
    </form>
    """


@app.route("/normalized_image_as_list", methods=["POST"])
def predict_normalized_image_as_list():
    artist_mapping = get_artist_dict()
    model = get_model()
    input_example = request.json["input_data"]
    np_input = np.array(input_example, dtype=np.float32)
    print(np_input.shape)
    tensor_input = torch.tensor(np_input).unsqueeze(0)
    logits = model(tensor_input)
    with torch.no_grad():
        prediction = int(torch.argmax(logits))
    str_prediction = str(prediction)
    return jsonify({"prediction": artist_mapping[str_prediction]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
