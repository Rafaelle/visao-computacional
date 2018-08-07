import cv2
import time
import os

def AddName():
    Name = input('Digite seu nome ')
    Info = open("Names.txt", "r+")
    ID = ((sum(1 for line in Info))+1)
    Info.write(str(ID) + "," + Name + "\n")
    print ("Nome armazenado como ID: " + str(ID))
    Info.close()
    return ID

def creatingDirectories():
    current_directory = os.getcwd()
    samplesDir = current_directory + '\samples'
    classifiersDir = current_directory + '\classifiers'

    if not os.path.isdir(samplesDir):
        os.makedirs(samplesDir)

    if not os.path.isdir(classifiersDir):
        os.makedirs(classifiersDir)

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # carregar arquivo
camera = cv2.VideoCapture(0) # capturar imagem da webcam 
amostra = 1 # controlar quantas fotos são tiradas na web (video)
numeroAmostras = 30 # valor ainda será estudado
# adicionar identificador para pessoas diferentes
creatingDirectories()
id = AddName()
largura, altura = 220, 220 # para normalizar as imagens devido aos algoritmos de reconhecimento, o tamanho devem ser iguais

print("Capturando faces...")

while (True) :
    conectcado, imagem = camera.read() # conectar e ler imagem da camera
    imagem = cv2.flip(imagem,180) #espelha a imagem

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # tranformando imagem vinda na web em tons de cinza 

    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = classificador.detectMultiScale(imagemCinza, 1.3, 5)

    # para imprimir o retangulo nas faces detectadas 
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 5) 
        # rectangle(imagem, inicio, final, rgb, borda)

        # quando a tecla Q for pressionada, capturar imagem
        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura)) # para ajustar ao tamanho que desejamos
            cv2.imwrite("samples/face_" + str(id) + "_" + str(amostra) + ".jpg", imagemFace)
            print("Amostra " + str(amostra) + " capturada")
            amostra += 1
           

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    # quando chegar no número de amostras para o programa
    if (amostra >= numeroAmostras + 1) : 
        break

print("Finalizado!")
camera.release() # liberar a memoria
cv2.destroyAllWindows() # fechar todas as janelas