from flask import Flask, render_template, request, json, jsonify
from fastai.vision.image import open_image, image2np
from PIL import Image as PILImage
import base64
from io import BytesIO

from fastai import *
from fastai.vision import *


app = Flask(__name__)

@app.route("/ping")
def hello(): 
  return "success"

_learner = None

@app.route('/')
def upload():
  return render_template('upload.html')

def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def _predict_single(fp):
  img = open_image(fp)
  pred_class,pred_idx,outputs = _learn.predict(img)
  #pred = img.predict(_learn)
  #idx = pred.argmax()
  #label = _learn.data.classes[idx]
  img_data = encode(img)
  #return { 'label': label, 'name': fp.filename, 'image': img_data }
  return { 'label': str(pred_class), 'name': fp.filename, 'image': img_data }


@app.route('/predict', methods=['POST'])
def predict():
  global _learn
  files = request.files.getlist("files")
  predictions = map(_predict_single, files)
  return render_template('predict.html', predictions=predictions)

@app.route('/predictData', methods=['POST'])
def predictData():
  global _learn
  path='/home/chihchungwang/Documents/git/github/chwang733/fastai-v3-examples/ComputerVision/data/bears'
  classes = ['black', 'grizzly', 'teddys']
  data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
  learn = create_cnn(data2, models.resnet34)
  learn.load('stage-2')
  _learn=learn

  files = request.files.getlist("files")
  fp=files[0]
  predictions = _predict_single(fp)  
  #print(predictions)
  return jsonify(predictions)

def serve(learn, port=9999):
    global _learn
    _learn = learn
    app.run(host='0.0.0.0',port=port)