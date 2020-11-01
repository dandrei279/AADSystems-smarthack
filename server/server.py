from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug import secure_filename
import os
import tempfile
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import json

def updateTags(decoded):
    a = ""
    for i in decoded:
        if i == "palace":
            a = a + " pilastri,"
        if i == "monastery":
            a = a + " ordine clasice,"
        if i == "church":
            a = a + " acoperis tabla,"
        if i == "triumphal arch":
            a = a + " coloane,"
        if i == "vault":
            a = a + " tencuiala,"
        if i == "altar":
            a = a + " inspiratie europeana,"
        if i == "street sign":
            a = a + " brutalism,"
        if i == "crane":
            a = a + " inspiratie sovietica,"
        if i == "flagpole":
            a = a + " functionalism,"
        if i == "pole":
            a = a + " ,"
        if i == "tobacco shop":
            a = a + " ,"
        if i == "mobile home":
            a = a + " neomodernism,"
        if i == "boathouse":
            a = a + " culori,"
        if i == "microwave":
            a = a + " texturi diverse,"
        if i == "solar dish":
            a = a + " neunitate stilistica,"
        if i == "stove":
            a = a + " sticla,"
        if i == "recreational vehicle":
            a = a + " metal,"
        if i == "ashcan":
            a = a + " structuri speciale,"
        if i == "window screen":
            a = a + " lumina,"
    return a

def getStyle (decoded):
    if np.array_equal(decoded,['mobile home','boathouse', 'recreational vehicle', 'microwave', 'solar dish']):
        return "Contemporan"
    if np.array_equal(decoded,['mobile home', 'stove', 'window screen', 'boathouse', 'ashcan']):
        return "Contemporan"
    if np.array_equal(decoded,['street sign', 'crane', 'flagpole' ,'pole' ,'tobacco shop']):
        return "Socialist"
    if np.array_equal(decoded,['palace' ,'monastery' ,'church', 'triumphal arch', 'vault']):
        return "Eclectic"
    if np.array_equal(decoded,['palace', 'vault' ,'monastery', 'triumphal arch', 'altar']):
        return "Eclectic"


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/ai-compare', methods = ['GET'])
@cross_origin()
def processFile():
    tmpdir = tempfile.mkdtemp()
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # file = tf.keras.utils.get_file("grace_hopper.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
    # img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])

    #Eclectic 2 = Eclectic 3
    img1 = tf.keras.preprocessing.image.load_img("./uploads/uploaded-photo_1.jpeg", target_size=[224, 224])
    #['palace' 'monastery' 'church' 'triumphal arch' 'vault']
    img2 = tf.keras.preprocessing.image.load_img("./uploads/uploaded-photo_2.jpeg", target_size=[224, 224])
    #['palace' 'vault' 'monastery' 'triumphal arch' 'altar']

    #img = tf.keras.preprocessing.image.load_img("C:\\Users\\DragosCristache\\Desktop\\socialist3.jpeg", target_size=[224, 224])
    # ['street sign' 'crane' 'flagpole' 'pole' 'tobacco shop']

    #img = tf.keras.preprocessing.image.load_img("C:\\Users\\DragosCristache\\Desktop\\contemporan1.jpeg", target_size=[224, 224])
    #['mobile home' 'boathouse' 'recreational vehicle' 'microwave' 'solar dish']
    #img = tf.keras.preprocessing.image.load_img("C:\\Users\\DragosCristache\\Desktop\\house3.jpeg", target_size=[224, 224])
    #['mobile home' 'stove' 'window screen' 'boathouse' 'ashcan']

    plt.imshow(img1)
    plt.axis('off')
    x1 = tf.keras.preprocessing.image.img_to_array(img1)
    x1 = tf.keras.applications.mobilenet.preprocess_input(x1[tf.newaxis, ...])
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels1 = np.array(open(labels_path).read().splitlines())
    pretrained_model1 = tf.keras.applications.MobileNet()
    result_before_save1 = pretrained_model1(x1)
    decoded1 = imagenet_labels1[np.argsort(result_before_save1)[0, ::-1][:5] + 1]

    plt.imshow(img2)
    plt.axis('off')
    x2 = tf.keras.preprocessing.image.img_to_array(img2)
    x2 = tf.keras.applications.mobilenet.preprocess_input(x2[tf.newaxis, ...])
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
    imagenet_labels2 = np.array(open(labels_path).read().splitlines())
    pretrained_model2 = tf.keras.applications.MobileNet()
    result_before_save2 = pretrained_model2(x2)
    decoded2 = imagenet_labels2[np.argsort(result_before_save2)[0, ::-1][:5] + 1]

    tags1 = updateTags(decoded1)
    tags2 = updateTags(decoded2)
    style1 = getStyle(decoded1)
    style2 = getStyle(decoded2)
    
    json_ = json.dumps({"tags1" : tags1, "style1" : style1, "tags2" : tags2, "style2":style2, "similar": style1 == style2})

    response = app.response_class(
        response=json_,
        status=200,
        mimetype='application/json'
    )
    return response
	
@app.route('/upload1', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      s = secure_filename(f.filename)
      f.save("./uploads/uploaded-photo_1." + s.split('.')[1])
      return 'file uploaded successfully', 200

@app.route('/upload2', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      f = request.files['file']
      s = secure_filename(f.filename)
      f.save("./uploads/uploaded-photo_2." + s.split('.')[1])
      return 'file uploaded successfully', 200
		
if __name__ == '__main__':
   app.run()

