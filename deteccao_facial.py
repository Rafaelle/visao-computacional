import cv2

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # carregar arquivo
camera = cv2.VideoCapture(0) # capturar imagem da webcam 
amostra = 1 # controlar quantas fotos são tiradas na web (video)
numeroAmostras = 10 # valor ainda será estudado
# adicionar identificador para pessoas diferentes

print("Capturando...")

while (True) :
    conectcado, imagem = camera.read() # conectar e ler imagem da camera
    imagem = cv2.flip(imagem,180) #espelha a imagem

    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # tranformando imagem vinda na web em tons de cinza 

    # detectar face em imagem cinza aumenta o desempenho
    facesDetectadas = classificador.detectMultiScale(imagemCinza, 
    scaleFactor=1.5,
    minSize = (100,100))

    ''' 
        facesDetectadas é uma matriz com posição (x,y), largura e altura das faces detectadas
        l = largura
        a = altura
        x,y = posição de inicio da face
    '''
    # para imprimir o retangulo nas faces detectadas 
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 5) 
        # rectangle(imagem, inicio, final, rgb, borda)

        # quando a tecla Q for pressionada, capturar imagem
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("fotos/face_" + str(amostra) + ".jpg", cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY))
            print("Foto " + str(amostra) + " capturada")
            amostra += 1

    cv2.imshow("Face", imagem)
    # quando chegar no número de amostras para o programa
    if (amostra >= numeroAmostras + 1) : 
        break     

print ("Finalizado!")
camera.release() # liberar a memoria
cv2.destroyAllWindows() # fechar todas as janelas