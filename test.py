import numpy as np
import cv2
import time

def AddName():
    Name = input('Digite seu nome ')
    Info = open("Names.txt", "r+")
    ID = ((sum(1 for line in Info))+1)
    Info.write(str(ID) + "," + Name + "\n")
    print ("Nome armazenado como ID: " + str(ID))
    Info.close()
    return ID

classificador = cv2.CascadeClassifier("C:\\Users\\carva\\Documents\\visao-computacional\\haarcascade_frontalface_default.xml") # carregar arquivo

amostra = 1 # controlar quantas fotos são tiradas na web (video)

numeroAmostras = 30 # valor ainda será estudado

# adicionar identificador para pessoas diferentes
id = AddName()

largura, altura = 220, 220 # para normalizar as imagens devido aos algoritmos de reconhecimento, o tamanho devem ser iguais

print("Capturando faces...")

camera = cv2.VideoCapture(0)
if camera.isOpened():
    while True:
        now = time.time()
        conectado, imagem = camera.read()

        imagem = cv2.flip(imagem,180) #espelha a imagem

        imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # tranformando imagem vinda na web em tons de cinza
        
        
        cv2.imshow('Color Frame', imagem)

        facesDetectadas = classificador.detectMultiScale(imagemCinza, 1.3, 5)
        
        key = cv2.waitKey(50)
        if key == ord('q'):
            break
            
        else:
            print('Rodando')
            
