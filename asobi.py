import os
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain


def check(a,b) :
    return a[2]*a[3] > b[2]*b[3]

def trimingFace() :

    DIR_NAME = "./material/"
    CASCADE_PARH = './anime.xml'

    files = os.listdir(DIR_NAME)
    cascade = cv2.CascadeClassifier(CASCADE_PARH)
    color = (255,255,255)
    faces = []

    for index, file in enumerate(files) :
        if(file == ".DS_Store") :
            continue

        img = cv2.imread(DIR_NAME+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

        if len(face_rects) > 0:
            face = [0,0,0,0]
            for face_rect in face_rects:
                if(check(face_rect, face)) :
                    face = face_rect
            newFace = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        faces.append(newFace)
        cv2.imshow("顔",newFace)
        
        cv2.waitKey()
        cv2.destroyAllWindows()
    return faces


def trimigHair(faces) :
    return [face[0:int(face.shape[0]/4),:] for face in faces]

def resize(hairs) :
    return [cv2.resize(hair,(int(hair.shape[1]/10), int(hair.shape[0]/10)))for hair in hairs]

def extractHairColor(hairs) :
    ret = []
    km = KMeans(n_clusters=2)
    for hair in hairs :
        color = []
        for h in hair :
            color.extend(h)
        a = km.fit(color)
        zeroNum = len([_a for _a in a.labels_ if _a==0])
        oneNum = len([_a for _a in a.labels_ if _a==1])

        check = 0 if zeroNum >= oneNum else 1
        cent0 = a.cluster_centers_[0]
        cent1 = a.cluster_centers_[1]
        for c, i in zip(color, a.labels_) :
            if(i==check) :
                dist0 = (c[0]-cent0[0])**2+(c[1]-cent0[1])**2+(c[2]-cent0[2])**2
                dist1 = (c[0]-cent1[0])**2+(c[1]-cent1[1])**2+(c[2]-cent1[2])**2
                if(dist0 < dist1) :
                    ret.append(cent0)
                else :
                    ret.append(cent1)
                break
    return ret

def clusteringFaces(colors, faces) :

    km = KMeans(n_clusters=3)
    result = km.fit(colors)
    print(result.labels_)
    #showFigure(result.labels_, colors)
    showImages(result.labels_, colors, faces)

def showFigure(labels, colors) :
    fig = plt.figure()
    ax = Axes3D(fig)
    color=["r", "b", "g"]
    for l,c in zip(labels, colors) :
        ax.scatter(c[0], c[1], c[2], color=color[l])

    plt.show()

def showImages(labels, colors, faces) :
    for i,l in enumerate(labels) :
        cv2.imshow(str(l),faces[i])

        cv2.waitKey()
        cv2.destroyAllWindows()



if __name__ == '__main__' :
    faces = trimingFace()
    hairs = trimigHair(faces)
    resizedHairs = resize(hairs)
    hairColors = extractHairColor(resizedHairs)
    clusteringFaces(hairColors, faces)
