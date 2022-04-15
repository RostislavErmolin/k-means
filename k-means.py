import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
from math import  sqrt
import random
import math
from PIL import Image
from PIL import ImageColor
img_path = input("Введите название изображения:")
K =  int(input("Введите количество кластеров:"))
original_image = cv2.imread(img_path)
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)


##
Point = namedtuple('Point', ('coords', 'n', 'ct'))
Cluster = namedtuple('Cluster', ('points', 'center', 'n'))


#imga = Image.open(filename)
def get_points(img):
    points = []
    w, h = img.size
    for count, color in img.getcolors(w * h):
        points.append(Point(color, 3, count))
    return points

rtoh = lambda rgb: '#%s' % ''.join(('%02x' % p for p in rgb))

def colorz(filename, n=K):
    img = Image.open(filename)
    img.thumbnail((200, 200))
    w, h = img.size

    points = get_points(img)
    clusters = kmeans(points, n, 1)
    rgbs = [map(int, c.center.coords) for c in clusters]
    return (list(map(rtoh, rgbs)))

def euclidean(p1, p2):
    return sqrt(sum([(p1.coords[i] - p2.coords[i]) ** 2 for i in range(p1.n)]))

def calculate_center(points, n):
    vals = [0.0 for i in range(n)]
    plen = 0
    for p in points:
        plen += p.ct
        for i in range(n):
            vals[i] += (p.coords[i] * p.ct)
    return Point([(v / plen) for v in vals], n, 1)

def kmeans(points, k, min_diff):
    clusters = [Cluster([p], p, p.n) for p in random.sample(points, k)]
    while 1:
        plists = [[] for i in range(k)]

        for p in points:
            smallest_distance = float('Inf')
            for i in range(k):
                distance = euclidean(p, clusters[i].center)
                if distance < smallest_distance:
                    smallest_distance = distance
                    idx = i
            plists[idx].append(p)

        diff = 0
        for i in range(k):
            old = clusters[i]
            center = calculate_center(plists[i], old.n)
            new = Cluster(plists[i], center, old.n)
            clusters[i] = new
            diff = max(diff, euclidean(old.center,new.center))
        if diff < min_diff:
            break
    return clusters
color = colorz(img)
list_color = list()
for row in list(color):
     hp =  list(ImageColor.getcolor(row, "RGB"))
     list_color.append(hp)
list_color = np.uint8(list_color) 
res = list_color[Point.flatten()]
result_image = res.reshape((img.shape))
figure_size = 10

#--Вывод
ax = plt.subplot

plt.figure('Сжатие цветов изображения методом кластеризации K-means',figsize=(figure_size,figure_size))
ax(2,2,1),plt.imshow(img)
plt.title('Оригинальное изображение'), plt.xticks([]), plt.yticks([])
ax(2,2,2),plt.imshow(result_image)
plt.title('Сегментированное изображение, когда K = %i' % K), plt.xticks([]), plt.yticks([])
k = 0
x_cord = 0
if K != 0:
     size_color = 1278/K 
for row in list_color:
     k = k + 1
     b = cv2.rectangle(img, (0+x_cord, 0), (int(size_color)*k, 900), (int(row[0]),int(row[1]),int(row[2])),-1)
     ax(2,2,(3,4)),plt.imshow(b)
     x_cord = int(size_color)*k
plt.title('Доменирующие цвета:'), plt.xticks([]), plt.yticks([])
plt.show()
