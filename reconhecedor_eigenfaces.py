import cv2

def FileRead():
    Info = open("Names.txt", "r")  # Abre o arquivo de texto com os nomes
    NAME = []
    while (True):  # Faz a leitura de todas as linhas e para quando não houver mais
        Line = Info.readline()
        if Line == '':
            break
        NAME.append(Line.split(",")[1].rstrip())

    return NAME  # Retorna a lista de nomes cadastrados

Names = FileRead()  # Executa a função para pegar os IDs e os nomes dos usuários

def ID2Name(ID):
    if ID > 0:
        NameString = Names[ID-1] # Busca o nome usando o indice do ID
    else:
        NameString = " Face não reconhecida "

    return NameString


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

while (True):
    conectado, imagem = camera.read()
    imagem = redim(imagem, 640)
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, 1.3, 5)

    frame_temp = imagem.copy()

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l],
                                (largura, altura))  # para ajustar ao tamanho que desejamos

        cv2.rectangle(frame_temp, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # desenhar o retangulo na imagem no ponto inicial (x,y) até o ponto final (x+l, y+a)
        id, confianca = reconhecedor.predict(imagemFace)

        name = ID2Name(id)

        # id = o nomedo objeto que conseguiu identificar
        cv2.putText(frame_temp, name, (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(frame_temp, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow("Face", redim(frame_temp, 640))
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
