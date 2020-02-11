import cv2 
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


I=cv2.imread('IMD002.bmp')
tamaño=I.shape
print (tamaño)
cv2.imshow('I',I)
cv2.waitKey(0)

#Filtro de mediana 
If=cv2.medianBlur(I,5)
#cv2.imshow('If',If)
#cv2.waitKey(0)
#Dull Razor

Ifg=cv2.cvtColor(If,cv2.COLOR_BGR2GRAY)
#cv2.imshow('Ifg',Ifg)
#cv2.waitKey(0)
"""k=np.ones((11,11),np.uint8)
Ic=cv2.morphologyEx(Ifg,cv2.MORPH_CLOSE,k)
cv2.imshow('Ic',Ic)
cv2.waitKey(0)"""
k=cv2.getStructuringElement(1,(11,11))
d=cv2.morphologyEx(Ifg,cv2.MORPH_BLACKHAT,k)
#cv2.imshow('diferencia',d)
#cv2.waitKey(0)
#diferencia de imagenes Ifg-Ic
"""d=cv2.absdiff(Ifg,Ic)
cv2.imshow('diferencia',d)
cv2.waitKey(0)"""
k1=np.ones((3,3),np.uint8)
Idc=cv2.morphologyEx(d,cv2.MORPH_CLOSE,k1)
#cv2.imshow('Idc',Idc)
#cv2.waitKey(0)
#Binarizacion
_,Ib=cv2.threshold(Idc,10,255,cv2.THRESH_BINARY)
#cv2.imshow('Mask',Ib)
#cv2.waitKey(0)
Ir=cv2.inpaint(I,Ib,8,cv2.INPAINT_TELEA)
cv2.imshow('I Dull Razor',Ir)
cv2.waitKey(0)

# Tomar regiones(Atenuar sombras)
"""Ihsv=cv2.cvtColor(Ir,cv2.COLOR_BGR2HSV)
(H,S,V)=cv2.split(Ihsv)
cv2.imshow('V',V)
cv2.waitKey(0)
S1=V[0:round(0.3*tamaño[0]),0:round(0.3*tamaño[1])]
S2=V[0:round(0.3*tamaño[0]),(tamaño[1]-round(0.3*tamaño[1])):tamaño[1]]
S3=V[(tamaño[0]-round(0.3*tamaño[0])):tamaño[0],0:round(0.3*tamaño[1])]
S4=V[(tamaño[0]-round(0.3*tamaño[0])):tamaño[0],(tamaño[1]-round(0.3*tamaño[1])):tamaño[1]]
Sh1=cv2.hconcat([S1,S2])
Sh2=cv2.hconcat([S3,S4])
SI=cv2.vconcat([Sh1,Sh2])
cv2.imshow('SI',SI)
cv2.waitKey(0)"""
def recorte(Iin):
	l=Iin.shape
	t=(round(l[0]-20),round(l[1]-20))
	c=(round(l[0]/2),round(l[1]/2))
	print(c)
	Io=cv2.getRectSubPix(Iin,t,c)
	cv2.imshow('Recorte',Io)
	cv2.waitKey(0)
def borde(Img):
	Iborde=cv2.Canny(Img,100,125)
	(cont,_)=cv2.findContours(Iborde.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(Img,cont,-1,(0,255,0),2)
	cv2.imshow('contornos',Img)
	cv2.waitKey(0)

def segtcol(Is):
	"""Falta incluir algo para mejorar el contraste
	suavizar"""
	(B,G,R)=cv2.split(Is)
	Ixyz=cv2.cvtColor(Is,cv2.COLOR_BGR2XYZ)
	(X,Y,Z)=cv2.split(Ixyz)
	Rn=cv2.equalizeHist(R)
	Xn=cv2.equalizeHist(X)
	Yn=cv2.equalizeHist(Y)
	_,Rot=cv2.threshold(Rn,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	_,Xot=cv2.threshold(Xn,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	_,Yot=cv2.threshold(Yn,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	"""print(Rot.shape)
	cv2.imshow('Rot',Rot)
	cv2.waitKey(0)
	cv2.imshow('Xot',Xot)
	cv2.waitKey(0)
	cv2.imshow('Yot',Yot)
	cv2.waitKey(0)"""
	Ipr=cv2.bitwise_and(Rot,Rot,mask=Xot)
	Ipr1=cv2.bitwise_and(Ipr,Ipr,mask=Yot)
	Ipr2=cv2.bitwise_and(Is,Is,mask=Ipr1)
	cv2.imshow('SegColor',Ipr2)
	cv2.waitKey(0)
#para extraer muestra
xi=round(tamaño[0]/2 -0.2*tamaño[0]/2)
xf=round(tamaño[0]/2 + 0.2*tamaño[0]/2)
yi=round(tamaño[1]/2 -0.2*tamaño[1]/2)
yf=round(tamaño[1]/2 + 0.2*tamaño[1]/2)
print(xi)
print(xf)
print(yi)
print(yf)
muestra=Ir[xi:xf,yi:yf]
cv2.imshow('muestra',muestra)
cv2.waitKey(0)
#Distancia de mahalanobis
def dmh(imi,mmed,mcovn):
	s=imi.shape
	mmed=mmed.astype(np.double)
	print('tamaño muestra')
	print(mmed.shape)
	dm2=np.zeros(s)
	for i in range(s[0]):
		for j in range(s[1]):
			z=np.array([imi[i,j,2],imi[i,j,1],imi[i,j,0]])
			z=z.astype(np.double)
			res=np.array([(z[0]-mmed[0]),(z[1]-mmed[1]),(z[2]-mmed[2])])
			dm2[i,j]=np.dot(np.dot(res.T,mcovn),res)
	print('dm2')
	print(dm2)
	print(dm2.shape)
	#Normalizar dm2
	dm2=dm2/255
	print('dm2 Normalizado')
	print(dm2)
	#sgdmh=np.zeros(dm2.shape)
	#np.where(dm2<2,dm2,0*dm2)
	dm2[dm2<0.28]=0
	print('segmetada')
	print(dm2)
	sgt=dm2.astype(np.float32)

	cv2.imshow('segmentacion mahala',sgt)
	cv2.waitKey(0)


#Segmentación por distancia de maholanobis
def distmalh(imo,imues):
	imues=imues.astype(np.double)
	Bm=imues[:,:,0]
	Gm=imues[:,:,1]
	Rm=imues[:,:,2]
	#conmues=np.concatenate((Rm,Gm,Bm),axis=0)
	fila=Rm.size
	Ra=np.ravel(Rm,order='C')
	Ga=np.ravel(Gm,order='C')
	Ba=np.ravel(Bm,order='C')
	Rb=Ra.reshape(fila,1)
	Gb=Ga.reshape(fila,1)
	Bb=Ba.reshape(fila,1)

	conmues=np.concatenate((Rb,Gb,Bb),axis=1)
	tm=conmues.shape
	covarc=np.cov(conmues,rowvar=False)
	invcovarc=np.linalg.inv(covarc)
	maxicv=np.max(invcovarc)
	invcn=invcovarc/maxicv
	#media de la muestra
	Rd=np.mean(Rm)
	Gd=np.mean(Gm)
	Bd=np.mean(Bm)
	medmues=np.array([Rd,Gd,Bd])
	print('covarianza por cov()')
	print(covarc)
	print(covarc.shape)
	print('covarianza inversa')
	print(invcovarc)
	print(invcovarc.shape)
	print('max covarianza inversa')
	print(maxicv)
	print('covarianza inversa normalizada')
	print(invcn)
	print('media de la muestra')
	print(medmues)
	print(medmues.shape)
	#Calculo de la distancia mahal...
	dmh(imo,medmues,invcovarc)


distmalh(Ir,muestra)