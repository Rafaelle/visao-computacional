import cv2

def redim(img, largura):  # função para redimensionar uma imagem
    alt = int(img.shape[0] / img.shape[1] * largura)
    img = cv2.resize(img, (largura, alt), interpolation=cv2.INTER_AREA)
    return img

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classifiers/classificadorEigen.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_SIMPLEX
camera = cv2.VideoCapture(0)  # id do dispositivo (camera principal do notebook)

pessoas = {
    0: "rafaelle",
    1: "josue",
    2: "tayna",
    3: "andre",
}

while (True):
    conectado, imagem = camera.read()
    imagem = redim(imagem, 640)
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.1,
                                                    minSize=(30, 30))

    frame_temp = imagem.copy()

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l],
                                (largura, altura))  # para ajustar ao tamanho que desejamos

        cv2.rectangle(frame_temp, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # desenhar o retangulo na imagem no ponto inicial (x,y) até o ponto final (x+l, y+a)
        id, confianca = reconhecedor.predict(imagemFace)
        # id = o nomedo objeto que conseguiu identificar
        cv2.putText(frame_temp, pessoas[id], (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(frame_temp, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow("Face", redim(frame_temp, 640))
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
