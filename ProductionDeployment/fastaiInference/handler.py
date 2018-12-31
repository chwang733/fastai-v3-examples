import json
from fastai.vision.image import open_image, image2np
from PIL import Image as PILImage
import base64
from io import BytesIO
#import boto3
import os

#from fastai import 
from fastai.vision import ImageDataBunch,create_cnn,models, get_transforms
from fastai.vision.data import imagenet_stats

HERE = os.path.dirname(os.path.realpath(__file__))

#path='/home/chihchungwang/Documents/git/github/chwang733/fastai-v3-examples/ComputerVision/data/bears'

path=os.path.join(HERE, "tmp")
'''
if os.path.isfile(path + '/models/00000014.jpg') != True:
    s3 = boto3.client('s3')
    s3.download_file('chihchung', 'public/fastai/teddy/models/00000014.jpg', path + '/models/00000014.jpg')


if os.path.isfile(path + '/models/stage-2.pth') != True:
    s3 = boto3.client('s3')
    s3.download_file('chihchung', 'public/fastai/teddy/models/stage-2.pth', path + '/models/stage-2.pth')
'''

classes = ['black', 'grizzly', 'teddys']
data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
_learn = create_cnn(data2, models.resnet34)
_learn.load('stage-2')


def encode(img):
    img = (image2np(img.data) * 255).astype('uint8')
    pil_img = PILImage.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def predict(event, context):

    global _learn
       
    img=open_image(path + '/models/00000014.jpg')

    pred_class,pred_idx,outputs = _learn.predict(img)
    img_data = encode(img)

    body = { 'label': str(pred_class), 'image': img_data }
    
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

if __name__ == "__main__":
    print(predict('', ''))

