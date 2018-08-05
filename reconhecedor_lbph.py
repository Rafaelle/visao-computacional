import cv2

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")
largura, altura = 220,220
font = cv2.FONT_HERSHEY_SIMPLEX
camera = cv2.VideoCapture(0) # id do dispositivo (camera principal do notebook)

pessoas = {
    0 : "rafaelle",
    1 : "josue",
    2 : "tayna",
    3 : "andre"
}


while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    
    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, 
    scaleFactor=1.5,
    minSize = (100,100))

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura)) # para ajustar ao tamanho que desejamos

        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0, 255), 2)
        #desenhar o retangulo na imagem no ponto inicial (x,y) até o ponto final (x+l, y+a)
        id, confianca = reconhecedor.predict(imagemFace)
        # id = o nomedo objeto que conseguiu identificar
        cv2.putText(imagem, pessoas[id], (x,y + (a + 30)), font, 3, (0,0,255))
        cv2.putText(imagem, str(confianca), (x, y + (a+50)), font, 1, (0,0,255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord("q"):
        break
        
camera.release()
cv2.destroyAllWindows()