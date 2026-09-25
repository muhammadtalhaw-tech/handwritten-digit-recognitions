from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import base64
import io

app = Flask(__name__)


# ============================================================
# LOAD BOTH MODELS
# ============================================================

MODEL_1_PATH = "mnist_handwritten_digit_model.keras"
MODEL_2_PATH = "mnist_handwritten_digit_fine_tuned.keras"

model1 = load_model(MODEL_1_PATH)
model2 = load_model(MODEL_2_PATH)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def prepare_image(image):

    # --------------------------------------------------------
    # 1. Convert image to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # Convert to numpy
    img = np.array(image)

    # --------------------------------------------------------
    # 2. Make digit white and background black
    # --------------------------------------------------------

    # Check corners
    corner_pixels = [
        img[0, 0],
        img[0, -1],
        img[-1, 0],
        img[-1, -1]
    ]

    corner_mean = np.mean(corner_pixels)

    # If background is white, invert image
    if corner_mean > 127:
        img = 255 - img

    # --------------------------------------------------------
    # 3. Find digit pixels
    # --------------------------------------------------------

    coords = np.argwhere(img > 30)

    # If nothing is written
    if coords.size == 0:

        return np.zeros(
            (1, 28, 28, 1),
            dtype="float32"
        )

    # --------------------------------------------------------
    # 4. Find bounding box
    # --------------------------------------------------------

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop only the digit
    digit = img[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------------
    # 5. Convert to PIL image
    # --------------------------------------------------------

    digit = Image.fromarray(
        digit.astype("uint8")
    )

    # --------------------------------------------------------
    # 6. Make image square
    # --------------------------------------------------------

    width, height = digit.size

    size = max(width, height)

    square = Image.new(
        "L",
        (size, size),
        0
    )

    left = (size - width) // 2
    top = (size - height) // 2

    square.paste(
        digit,
        (left, top)
    )

    # --------------------------------------------------------
    # 7. Resize digit to 20x20
    # --------------------------------------------------------

    digit = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # 8. Put digit in center of 28x28 image
    # --------------------------------------------------------

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    final_image.paste(
        digit,
        (4, 4)
    )

    # --------------------------------------------------------
    # 9. Convert to numpy
    # --------------------------------------------------------

    img = np.array(
        final_image
    ).astype("float32")

    # --------------------------------------------------------
    # 10. Normalize
    # --------------------------------------------------------

    img = img / 255.0

    # --------------------------------------------------------
    # 11. Reshape for MNIST model
    # --------------------------------------------------------

    img = img.reshape(
        1,
        28,
        28,
        1
    )

    return img


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        mouse_prediction1=None,
        mouse_prediction2=None,
        image_prediction1=None,
        image_prediction2=None,
        uploaded_image=None
    )


# ============================================================
# MOUSE DRAWING PREDICTION
# ============================================================

@app.route(
    "/predict_mouse",
    methods=["POST"]
)
def predict_mouse():

    try:

        # Get base64 image
        data = request.json["image"]

        # Remove image header
        data = data.split(",")[1]

        # Decode base64
        image_data = base64.b64decode(data)

        # Open image
        image = Image.open(
            io.BytesIO(image_data)
        )

        # Prepare image
        img = prepare_image(image)

        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        prediction1 = model1.predict(
            img,
            verbose=0
        )

        digit1 = int(
            np.argmax(prediction1)
        )

        confidence1 = float(
            np.max(prediction1) * 100
        )

        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        prediction2 = model2.predict(
            img,
            verbose=0
        )

        digit2 = int(
            np.argmax(prediction2)
        )

        confidence2 = float(
            np.max(prediction2) * 100
        )

        # Return results
        return jsonify({

            "success": True,

            "model1": digit1,

            "model2": digit2,

            "confidence1": round(
                confidence1,
                2
            ),

            "confidence2": round(
                confidence2,
                2
            )

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        })


# ============================================================
# IMAGE UPLOAD PREDICTION
# ============================================================

@app.route(
    "/predict_image",
    methods=["POST"]
)
def predict_image():

    try:

        # Get uploaded file
        file = request.files["image"]

        # Open image
        image = Image.open(file)

        # Prepare image
        img = prepare_image(image)

        # ----------------------------------------------------
        # MODEL 1
        # ----------------------------------------------------

        prediction1 = model1.predict(
            img,
            verbose=0
        )

        digit1 = int(
            np.argmax(prediction1)
        )

        confidence1 = float(
            np.max(prediction1) * 100
        )

        # ----------------------------------------------------
        # MODEL 2
        # ----------------------------------------------------

        prediction2 = model2.predict(
            img,
            verbose=0
        )

        digit2 = int(
            np.argmax(prediction2)
        )

        confidence2 = float(
            np.max(prediction2) * 100
        )

        # ----------------------------------------------------
        # Save uploaded image
        # ----------------------------------------------------

        os.makedirs(
            "static/uploads",
            exist_ok=True
        )

        filename = "uploaded_digit.png"

        filepath = os.path.join(
            "static/uploads",
            filename
        )

        image.save(filepath)

        # ----------------------------------------------------
        # Show results
        # ----------------------------------------------------

        return render_template(
            "index.html",

            mouse_prediction1=None,

            mouse_prediction2=None,

            image_prediction1=digit1,

            image_prediction2=digit2,

            image_confidence1=round(
                confidence1,
                2
            ),

            image_confidence2=round(
                confidence2,
                2
            ),

            uploaded_image=
            "/static/uploads/" + filename
        )

    except Exception as e:

        return render_template(
            "index.html",

            mouse_prediction1=None,

            mouse_prediction2=None,

            image_prediction1=None,

            image_prediction2=None,

            uploaded_image=None,

            error=str(e)
        )


# ============================================================
# RUN FLASK APP
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )