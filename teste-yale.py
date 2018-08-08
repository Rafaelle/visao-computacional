import cv2
import os
import numpy as np
from PIL import Image

def executeRecognizer(reconhecedor, nome):
    detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    totalAcertos = 0
    percentualAcerto = 0.0
    totalConfianca = 0.0

    print("\nAlgoritmo: " + nome)

    caminhos = [os.path.join('yalefaces/teste', f) for f in os.listdir('yalefaces/teste')]
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemFaceNP = np.array(imagemFace, 'uint8')
        facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP, scaleFactor=1.5, minSize=(100, 100))
        for (x, y, l, a) in facesDetectadas:
            idprevisto, confianca = reconhecedor.predict(imagemFaceNP)
            idatual = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
            print(str(idatual) + " foi classificado como " + str(idprevisto) + " - " + str(confianca))
            if idprevisto == idatual:
                totalAcertos += 1
                totalConfianca += confianca
            # cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
            # cv2.imshow("Face", imagemFaceNP)
            # cv2.waitKey(1000)
    percentualAcerto = (totalAcertos / 30) * 100
    totalConfianca = totalConfianca / totalAcertos
    print("Percentual de acerto: " + str(percentualAcerto))
    print("Total confiança: " + str(totalConfianca))


reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classifiers/classifiersYale/classificadorEigenYale.yml")
executeRecognizer(reconhecedor, "EigenFaces")

reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classifiers/classifiersYale/classificadorFisherYale.yml")
executeRecognizer(reconhecedor, "FisherFaces")

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classifiers/classifiersYale/classificadorLBPHYale.yml")
executeRecognizer(reconhecedor, "LBPH")