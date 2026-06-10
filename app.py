from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")

IMG_SIZE = 150

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        file = request.files["file"]
        filepath = "static/" + file.filename
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            result = "Tumor Detected"
        else:
            result = "No Tumor"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)