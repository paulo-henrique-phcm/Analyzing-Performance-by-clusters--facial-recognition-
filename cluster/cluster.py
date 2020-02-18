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
'''#---------------------------------------------------------------------------------------------------------------------------------------
vggface2 = VGGFace(model='resnet50', include_top=True)
vggface2.summary()
vggface2 = Model(inputs=vggface2.layers[0].input, outputs=vggface2.layers[-3].output)
'''

def preprocess_image(image_end):
    img = load_img(image_end, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def acessa_imgs(qtdPessoas, qtdImgs):
    #data = glob('/home/co/Documentos/vgg-face/face-data-padrao/*')
    predicts = []
    labels = []
    '''
    for pessoa, i in zip(data, range(qtdPessoas)):
        imagens = glob('/home/co/Documentos/vgg-face/face-data-padrao/'+os.path.split(pessoa)[-1]+'/*')
        for j, imagem in zip(range(qtdImgs), imagens):


            img = preprocess_image(imagem)
            proc = vggface2senet.predict(img)
            predicts.append(proc)
            labels.append(i)
            print(imagem)

    '''
    #vamos selecionar as primeiras "qtdPessoas" e delas, as primeiras "qtdImagens" de forma não aleatória
    pessoas = glob('/home/co/Documentos/vgg-face/face-data-padrao/*')
    meuVetorDePessoasComImagens = []
    for i, q in zip(pessoas, range(qtdPessoas)):
        imagens = glob(i+'/*')
        for j, q2 in zip(imagens, range(qtdImgs)):
            meuVetorDePessoasComImagens.append(j)

    print('quantidade de imagens preparadas: ',len(meuVetorDePessoasComImagens))
    print('extraindo caracteristicas')

    for i in meuVetorDePessoasComImagens:
        print('->',i)
        img = preprocess_image(i)
        proc = vggface2.predict(img)
        procRedimensionado = np.reshape(proc, (2048))
        predicts.append(procRedimensionado)
        #labels.append(i)
        #print(imagem)


    #print(imagens[1])
    #print(labels)
    #return predicts, labels
    return predicts, meuVetorDePessoasComImagens


'''
def load_plot():
    qtdImgs = 5
    qtdPessoas = 5


    predicts, labels = acessa_imgs(qtdImgs, qtdPessoas)
    print(len(predicts))
    print(len(labels))
    predicts = np.asarray(predicts)
    X = np.reshape(predicts, (len(labels), 2048))
    #KMeans.labels_array(labels, dtype=int32)
    kmeans = KMeans(n_clusters=qtdPessoas)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    print(X.shape)
    print(y_kmeans)

    np.save('/home/co/Documentos/vgg-face/face-data-padrao/y_kmeans.npy', y_kmeans)
    print(y_kmeans.shape)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.9)


    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

    #analise de consistencia

    a = np.reshape(y_kmeans(qtdPessoas,qtdImgs))
    print(a)
    return X, y_kmeans

'''

# Agora vamos tentar pegar algumas imagens, detectar as estouradas com o kameans de modo que vamos saber quais sao elas
def aparicoes_categoria(linha, categoria):
    #esta funcao retorna quantas vezes os elementos de uma linha se repetem numa determinada categoria
    aparicoes = 0
    for elemento in linha:
        if elemento == categoria:
            aparicoes = aparicoes + 1
    return aparicoes

def acertos():
    qtdPessoas = 50
    qtdImgs = 10
    X, enderecos = acessa_imgs(qtdPessoas,qtdImgs)
    #print(len(X))
    #print(len(X[1]))
    #X = np.load('/home/co/Documentos/vgg-face/vggface2_predicts_Grande/probes.npy')
    #Y = np.load('/home/co/Documentos/vgg-face/vggface2_predicts_Grande/classes.npy')


    '''
    mudancas = []
    mudancas.append(0)
    #aqui estou criando um vetor com as posições onde ocorre mudança de classi Y, ou de uma pessoa para outra

    for i in range(len(Y)-1):
        if (Y[i] != Y[i+1]):
            mudancas.append(i)
    mudancas.append(len(Y))
    '''
    #uns testes
    #print(len(X[1]))
    #print(Y)



    kmeans = KMeans(n_clusters=qtdPessoas)
    kmeans.fit(X)


    y_kmeans = kmeans.predict(X)

    for i in y_kmeans:
        print(i)

    PessoasH = []
    for i in range(qtdPessoas):
        IndividualH = []
        for j in range((i)*qtdImgs, (i+1)*qtdImgs):
            IndividualH.append(y_kmeans[j])
        PessoasH.append(IndividualH)

        ################################################################################################################################
        #aqui vamos identificar quais são as classes de cada linha para posteriormente sabermos o que está com problemas
        #para isso vamos percorrer todas as classes que conhecemos, e para cada uma, contar na matriz qual linha tem mais dessa classes
        #assim assumiremos que esta classe pertence áquela linha
    todas_categorias = []
    for categoria in range(qtdPessoas):
        probabilidade_caegoria = []

        for k in range(qtdPessoas):
            probabilidade_caegoria.append(aparicoes_categoria(PessoasH[k], categoria))
            #ao inves de adicionar, posso somar, pra saber em qual tem mais
        todas_categorias.append(probabilidade_caegoria)

    for i in range(qtdPessoas):
        print(PessoasH[i])
    for i in range(qtdPessoas):
        print(todas_categorias[i])

    #vamos procurar agora onde cada classe aparece mais, então ela pertencerá aquela linha

    np.save('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/PessoasH.npy', PessoasH)
    np.save('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/todas_categorias.npy', todas_categorias)
    np.save('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/enderecos.npy', enderecos)



    '''
    y_kmeans = kmeans.predict(X)
    vec = []


    print(mudancas)
    print(len(mudancas))
    h = 0

    kmeans_juntos = []
    mudancas = np.asarray(mudancas)
    for i in range(qtdPessoas):
        mesma = []
        kmeans_separado = []
        for j in range(mudancas[i], mudancas[i+1]):
            kmeans_separado.append(y_kmeans[j])
        h = h+len(mesma)
        vec.append(mesma)
        kmeans_juntos.append(kmeans_separado)
    kmeans_juntos = np.asarray(kmeans_juntos)
    print("qtd de pessoas: ",len(kmeans_juntos))
    print("qtd de amostras: ",h)
    '''

#acertos()

def procura_categorias(tudo):
    #passada a linha da classe, retorna a posicao em que aparece e quantidade de vezes que aparece
    posicaoClasses = np.zeros((len(tudo),), dtype=int)
    for i in range(len(tudo)):
        n_max = max(tudo[i])
        n_pos = list(tudo[i]).index(n_max)
        posicaoClasses[i] = n_pos
        print('classe ', i, 'aparece no maximo ', n_max, ' vezes na pasta: ', n_pos)


    return posicaoClasses

def imagensLugarErrado(possCateg, PessoasH):
    pastasNaoEncontradas = []
    for i in range(len(PessoasH)):
        print('a classe ', i, 'do k-means provavelmente esta na pasta', possCateg[i],'    ', PessoasH[possCateg[i]])
        if not i in possCateg:
            pastasNaoEncontradas.append(i)
    print('\nATENÇÃO  ->  as pastas ', pastasNaoEncontradas, 'não receberam nenhuma classificação (nao tem nenhuma classe). O código verifica onde cada classe aparece mais vezes, e assume que aquela classe pertence áquela pasta, pode ser que algumas classes estejam em mais de uma pasta e algumas delas fiquem sem classificação. Verifique-as a seguir')




#------------------------------------------------------------------------------------------------------------------------------------
PessoasH = np.load('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/PessoasH.npy')
todas_categorias = np.load('/home/co/Documentos/vgg-face/kmeans-face-dada-padrao/todas_categorias.npy')
qtdPessoas = 50


possivelSequenciaClasses = procura_categorias(todas_categorias)
print('\nOU SEJA:\n')
#print(possivelSequenciaClasses)

imagensLugarErrado(possivelSequenciaClasses, PessoasH)
print('\n\ncategorias na sequencia em que carregou das pastas: ')

print('pasta    categorias')
for i in range(len(PessoasH)):
    print(i, '-> ', PessoasH[i])




#vamos analisar o vetor categorias para ver onde cada categoria aparece maia

#y_kmeans = np.load('/home/co/Documentos/vgg-face/face-data-padrao/y_kmeans.npy')
#X = np.load('/home/co/Documentos/vgg-face/face-data-padrao/Xmeans.npy')
#X, y = load_plot()
