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

largura, altura = 220, 220
font = cv2.FONT_HERSHEY_SIMPLEX
camera = cv2.VideoCapture(0)  # id do dispositivo (camera principal do notebook)

#detector de face
detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# reconhecedores
reconhecedor_eigen = cv2.face.EigenFaceRecognizer_create()
reconhecedor_eigen.read("classifiers/classificadorEigen.yml")

reconhecedor_fisher = cv2.face.FisherFaceRecognizer_create()
reconhecedor_fisher.read("classifiers/classificadorFisher.yml")

reconhecedor_lbph = cv2.face.LBPHFaceRecognizer_create()
reconhecedor_lbph.read("classifiers/classificadorLBPH.yml")

def reconhecedor(imageFace, reconhecedor, frame, x, y, l, a ): # função para montar frame de um reconhecedor
    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # desenhar o retangulo na imagem no ponto inicial (x,y) até o ponto final (x+l, y+a)
        # (imagem, pos_inicial, pos_final, cor, borda)
    id, confianca = reconhecedor.predict(imagemFace)

    name = ID2Name(id)

    # id = o nome do objeto que conseguiu identificar
    cv2.putText(frame, name, (x, y + (a + 30)), font, 2, (0, 0, 255))
    cv2.putText(frame, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    return frame

print ("Iniciando reconhecimento...")
while (True):
    conectado, imagem = camera.read()
    imagem = redim(imagem, 640)
    imagem = cv2.flip(imagem,180) #espelha a imagem

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)

    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, 1.3, 5)

    frame_temp_eigen = imagem.copy()

    frame_temp_fisher = imagem.copy()

    frame_temp_lbph = imagem.copy()

    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l],
                                (largura, altura))  # para ajustar ao tamanho que desejamos

        ''' para a imagem com o eigen '''
        frame_temp_eigen = reconhecedor(imagemFace, reconhecedor_eigen, frame_temp_eigen, x, y, l, a)
      
        ''' para a imagem com fisher '''
        frame_temp_fisher = reconhecedor(imagemFace, reconhecedor_fisher, frame_temp_fisher, x, y, l, a)

        ''' para a imagem com LBPH '''
        frame_temp_lbph = reconhecedor(imagemFace, reconhecedor_lbph, frame_temp_lbph, x, y, l, a)

    cv2.imshow("Reconhecedor Eigen", frame_temp_eigen)
    cv2.imshow("Reconhecedor Fisher", frame_temp_fisher)
    cv2.imshow("Reconhecedor LBPH", frame_temp_lbph)


    if cv2.waitKey(1) == ord("q"):
        print("Finalizando reconhecimento...") 
        break

camera.release()
cv2.destroyAllWindows()
