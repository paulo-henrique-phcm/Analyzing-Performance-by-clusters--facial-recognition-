from sklearn.cluster import KMeans
import numpy as np
from keras_vggface.vggface import VGGFace
from glob import glob
from keras.models import Model, Sequential
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras_vggface import utils
import matplotlib.pyplot as plt
import os
from keras.layers import Conv2D, MaxPool2D, Input, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Activation, Flatten

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import matplotlib.image as mpimg
#---------------------------------------------------------------------------------------------------------------------------------------
PessoasH = np.load('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/PessoasH.npy')
End = np.load('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/enderecos.npy')

def mostraImagem(PessoasH, End, pasta):
    imgPasta = int((pasta*(len(End)/len(PessoasH))))

    plt.figure(figsize=(224, 224))
    for i in range(0, int((len(End)/len(PessoasH))/2)):
        plt.subplot(2,int((len(End)/len(PessoasH))/2),i+1)
        img = mpimg.imread(End[imgPasta+i])
        plt.title(str(i))
        plt.imshow(img)

    for i in range(int((len(End)/len(PessoasH))/2), int((len(End)/len(PessoasH)))):
        plt.subplot(2,int((len(End)/len(PessoasH))/2),i+1)
        img = mpimg.imread(End[imgPasta+i])
        plt.title(str(i))
        plt.imshow(img)
        #plt.set_title('axes title')

    plt.show()


print(len(End))

while True:
    pasta = input("Nº da pasta: ")
    pasta = int(pasta)
    print('acessando imagem...')
    mostraImagem(PessoasH, End, pasta)
