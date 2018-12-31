from flask import Flask
#from flask_cors import CORS, cross_origin
from flask import Response

import json
from fastai.vision.image import open_image, image2np
from PIL import Image as PILImage
import base64
from io import BytesIO
#import pickle
import os
#import numpy as np
#import torch 
#import boto
#from pytorch_models import awd_lstm

#from fastai import 
from fastai.vision import ImageDataBunch,create_cnn,models, get_transforms
from fastai.vision.data import imagenet_stats

app = Flask(__name__)
#CORS(app)

HERE = os.path.dirname(os.path.realpath(__file__))

bearspath=os.path.join(HERE, "tmp/bears")
classes = ['black', 'grizzly', 'teddys']

_bearslearn = create_cnn(ImageDataBunch.single_from_classes(bearspath, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats), models.resnet34)
_bearslearn.load('stage-2')

'''
if os.path.isfile(path + '/models/00000014.jpg') != True:
    s3 = boto3.client('s3')
    s3.download_file('chihchung', 'public/fastai/teddy/models/00000014.jpg', path + '/models/00000014.jpg')


if os.path.isfile(path + '/models/stage-2.pth') != True:
    s3 = boto3.client('s3')
    s3.download_file('chihchung', 'public/fastai/teddy/models/stage-2.pth', path + '/models/stage-2.pth')
'''


def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.route('/bears/inference',methods=['GET'])
def bearsInference():
     
    img=open_image(bearspath + '/models/00000014.jpg')

    pred_class,pred_idx,outputs = _bearslearn.predict(img)
    img_data = encode(img)

    body = { 'label': str(pred_class), 'image': img_data }
    
    resp= Response(response=json.dumps({"response": body}), status=200, mimetype='application/json')

    return resp


IS_LOCAL = False
if __name__ == '__main__':
    IS_LOCAL = True
    app.run(debug=True, host='0.0.0.0', port=8082)

#if __name__ == "__main__":
#    print(bearsInference())

