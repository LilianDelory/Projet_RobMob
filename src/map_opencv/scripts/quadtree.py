#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib as plt

# Fonction pour vérifier si tous les pixels dans une zone sont de la même couleur
def same_color(image):
    print(image.shape)
    print(image[0, 0])
    print(np.all(image == image[0, 0]))
    print("\n")
    return np.all(image == image[0, 0])

# Fonction pour diviser une zone en quatre quadrants
def split(image):
    h, w = image.shape
    return [image[:h//2, :w//2], image[:h//2, w//2:], image[h//2:, :w//2], image[h//2:, w//2:]]

# Fonction pour construire le quadtree
def build_quadtree(image):
    if same_color(image):
        
        return [image[0, 0]]
    else:
        return [build_quadtree(subimage) for subimage in split(image)]

# Fonction pour afficher la structure de l'arbre
def print_quadtree(quadtree, depth=0):
    if isinstance(quadtree, list):
        print("Depth:", depth)
        for i, subquad in enumerate(quadtree):
            print("Subquad", i, ":")
            print_quadtree(subquad, depth + 1)
    else:
        print("Color:", quadtree)  

# Votre image binaire 800x800

# Fonction pour afficher graphiquement la structure de l'arbre
def plot_quadtree(ax, quadtree, x, y, size):
    if isinstance(quadtree, list):
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, edgecolor='black'))
        size /= 2
        plot_quadtree(ax, quadtree[0], x, y, size)
        plot_quadtree(ax, quadtree[1], x + size, y, size)
        plot_quadtree(ax, quadtree[2], x, y + size, size)
        plot_quadtree(ax, quadtree[3], x + size, y + size, size)
    else:
        color = quadtree[0]
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=True, color=str(color / 255.0)))



image_path = "map.png"
binary_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)


_, binary_image = cv2.threshold(binary_image,128,255,cv2.THRESH_BINARY)


#Construction du quadtree
quadtree = build_quadtree(binary_image)
print_quadtree(quadtree)

# Affichage graphique de la structure de l'arbre
fig, ax = plt.subplots()
ax.set_xlim(0, binary_image.shape[1])
ax.set_ylim(0, binary_image.shape[0])
plot_quadtree(ax, quadtree, 0, 0, binary_image.shape[1])

plt.show()
