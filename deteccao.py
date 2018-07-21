import cv2

# detecção simples
def calculaDiferenca(img1, img2, img3):
  d1 = cv2.absdiff(img3, img2)
  d2 = cv2.absdiff(img2, img1)
  return cv2.bitwise_and(d1, d2)

# adicionando a operação de threshold para aumentar a diferença dos valores 
def calculaDiferenca2(img1, img2, img3):
  d1 = cv2.absdiff(img3, img2)
  d2 = cv2.absdiff(img2, img1)
  imagem = cv2.bitwise_and(d1, d2)
  s,imagem = cv2.threshold(imagem, 35, 255, cv2.THRESH_BINARY)
  return imagem


webcam = cv2.VideoCapture(0) #instancia o uso da webcam
janela = "Tela de captura"
cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE) #cria uma janela

#faz a leitura inicial de imagens
ultima        = cv2.cvtColor(webcam.read()[1], cv2.COLOR_RGB2GRAY) # a leitura da imagem corrente em tons de cinza
penultima     = ultima
antepenultima = ultima

#faz a leitura recorente das imagens
while True:
    # atualização das imagens
  antepenultima = penultima
  penultima     = ultima
  ultima        = cv2.cvtColor(webcam.read()[1], cv2.COLOR_RGB2GRAY)

  cv2.imshow(janela, calculaDiferenca2(antepenultima,penultima,ultima))
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyWindow(janela)
    break

print ("Fim")