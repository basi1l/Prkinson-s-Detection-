from flask import Flask ,render_template,request
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import resnet50
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model



mmodel = pickle.load(open('mmodel.pkl','rb'))

enet_bmodel =efficientnet.EfficientNetB1(input_shape=(224,224,3),include_top=False,weights='imagenet')
vgg_bmodel =VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')
rnet_bmodel =resnet50.ResNet50(input_shape=(224,224,3),include_top=False,weights='imagenet')

vgg_model = Model(inputs=vgg_bmodel.layers[0].input,outputs=vgg_bmodel.layers[-1].output)
# resenet
rnet_model = Model(inputs=rnet_bmodel.layers[0].input,outputs=rnet_bmodel.layers[-1].output)
# efficient net
enet_model = Model(inputs=enet_bmodel.layers[0].input,outputs=enet_bmodel.layers[-1].output)

def preprocess(im):
  H = median(im)
  # H = gaussfilt(H)
  H = equalize_adapthist(H)
  H = resize(H,[224,224])
  return H

def feature_extraction(inp):
  inp1 = np.expand_dims(inp,axis=0)
  fet1 = vgg_model.predict(inp1)
  fet1 = np.array(fet1).flatten().reshape(49,512)
  pca = PCA(n_components=20)
  nfet1 = pca.fit_transform(fet1).flatten()

  fet2 = rnet_model.predict(inp1)
  fet2 = np.array(fet2).flatten().reshape(49,2048)
  nfet2 = pca.fit_transform(fet2).flatten()

  fet3 = enet_model.predict(inp1)
  fet3 = np.array(fet3).flatten().reshape(49,1280)
  nfet3 = pca.fit_transform(fet3).flatten()
  fet =np.concatenate([nfet1,nfet2,nfet3])
  return fet

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html",predictions = None)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No files Uploaded'
    
    file = request.files['file']
    filename = file.filename
    imgpath = 'static/'+filename
    file.save(imgpath)
    time.sleep(2)
    print('image saved')
    img = plt.imread(imgpath)
    img = preprocess(img)
    f = feature_extraction(img)
    probs = mmodel.predict_proba([f])
    result = np.argmax(probs)
    prob = np.max(probs)
    print(prob)
    if prob<0.8:
        pred = 'None'
    else:
        if result:
            print('healthy')
            pred = 'healthy'
        else:
            print('parkinson')
            pred = 'parkinson'
    return render_template('home.html', prediction = pred, image = imgpath)
if __name__ == "__main__":
    app.run(debug=True)