import cv2
import os  # recursos do sistema operacional
import numpy as np

#criar os classificadores
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

# para percorrer as imagens salvas na captura (baseado no ID)


def getImagemComId():
    #lista com todas as fotos da pessoa
    #caminhos = []
    ids = []
    faces = []

    #tirar
    '''for f in os.listdir('fotos'):
        caminhos.append('fotos\\' + f)
        #print(f)
    print(caminhos)
'''
    #percorrendo todas as imagens
    for caminhoImag in os.listdir('samples'):
        imagemFace = cv2.cvtColor(cv2.imread(
            "samples\\"+caminhoImag), cv2.COLOR_BGR2GRAY)
        # para pegar o identificador de cada pessoa (verificar)
        id = int(os.path.split(caminhoImag)[-1].split('.')[1])
        #lista de ids
        ids.append(id)
        #lista de rostos
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
    #necessário para fazer o treinamento (ver depois)
    return np.array(ids), faces
    '''
    fotos = pasta
  '''


#matriz imagem e id
ids, faces = getImagemComId()

print("Treinando...")

#treinamento supervisionado -> passa as faces (entradas) e os ID's (saídas esperadas)
#ter pelo menos duas pessoas
#gravar na pasta a classificação dos registros

print("Treinando com eigenFace..")
eigenface.train(faces, ids)
eigenface.write("classifiers/classificadorEigen.yml")

print("Treinando com fisherFace..")
fisherface.train(faces, ids)
fisherface.write("classifiers/classificadorFisher.yml")

print("Treinando com LBPH..")
lbph.train(faces, ids)
lbph.write("classifiers/classificadorLBPH.yml")

print("Treinamento realizado")
