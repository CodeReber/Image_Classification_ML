import os
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing import image

# Hide CPU Tensorflow AVX Message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = keras.backend.clear_session()


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
graph = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(img):
    # Convert image to a numpy array
    img = image.img_to_array(img)

    img /= 255.0

    # Invert image pixels
    img = 1 - img

    # Flatten image to an array of pixels
    image_array = img.flatten().reshape(-1, 28*28)

    return image_array


def load_model():
    global model
    global graph
    model = keras.models.load_model("./models/mnist_trained.h5")
    graph = tf.Graph()


load_model()
## ================================ ##
## ====== FRONT END ROUTES ======== ##
## ================================ ##


@app.route('/home')
def home():
    return render_template("index.html")
## ================================ ##
## ====== BACK END ROUTES ======== ##
## ================================ ##


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_size = (28, 28)

            im = image.load_img(
                filepath, target_size=image_size, color_mode="grayscale")

            image_array = prepare_image(im)
            # return redirect(url_for('upload_file',
            # filename=filename))
            # print in terminal the array of image uploaded
            # print(image_array)

            return "Data Pre-Processing Complete!"

            # Tensorflow default graph and use to make predictions
            global graph
            with graph.as_default():
                predicted = model.predict_classes(image_array)[0]
                data["prediction"] = str(predicted)
                data["success"] = True

            return jsonify(data)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
