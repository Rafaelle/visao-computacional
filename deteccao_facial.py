import cv2

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # carregar arquivo
camera = cv2.VideoCapture(0) # capturar imagem da webcam 

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

    cv2.imshow("Face" , imagem) # mostrar imagem com face detectada

    #parar o código e fechar a janela
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



camera.release() # liberar a memoria
cv2.destroyAllWindows() # fechar todas as janelas