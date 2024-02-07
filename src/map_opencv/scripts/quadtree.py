#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

DEBUG = False

# Fonction qui teste si l'image que l'on regarde est de couleur unie.
# @image une image de taille carré
# retourne 1 si oui sinon 0
def same_color(image):
    return np.all(image == image[0, 0])

# Fonction qui decoupe une image en 4 images plus petites correspondant à chaque coin
# @image une image de taille carré
# retoune un tableau avec les 4 images dedans.
def split(image):
    h, w = image.shape
    return [image[:h//2, :w//2], image[:h//2, w//2:], image[h//2:, :w//2], image[h//2:, w//2:]]

# Fonction récursive qui permet de representer l'arbre d'un quadtree
# @image une image de taille carré
# @graph un graph (déjà formé ou non)
# @ parent : le parent du noeud
# @ edge_label : une etiquette d'arête
# retoune un tableau avec les 4 images dedans.
def build_quadtree_graph(image, graph=None, parent=None, edge_label=None):

    # Si le graphe n'a pas été construit alors on le construit.
    if graph is None:
        graph = nx.DiGraph()

    if same_color(image):
        color = image[0, 0]
        graph.add_node(color)
        # S'il y a un parent on ajoute une arête
        if parent is not None:
            graph.add_edge(parent, color, label=edge_label)
    else:
        node_label = str(image.shape)
        graph.add_node(node_label)
        if parent is not None:
            graph.add_edge(parent, node_label, label=edge_label)

        subimages = split(image)
        for i, subimage in enumerate(subimages):
            build_quadtree_graph(subimage, graph, node_label, i)

    return graph

# Fonction récursive pour construire le quadtree sur l'image d'origine
# @image une image de taille carré
def build_quadtree(image):
    if same_color(image):
        return image[0, 0]
    else:
        return [build_quadtree(subimage) for subimage in split(image)]


# Fonction pour afficher le graphe lié au quadtree
def plot_quadtree_graph(graph, pos=None):
    if pos is None:
        pos = graphviz_layout(graph, prog="dot")
    labels = nx.get_edge_attributes(graph, 'label')
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=8, font_color='black', font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

# Fonction pour afficher le quadtree sur l'image d'origine
def plot_quadtree(ax, quadtree, x, y, size):
    if DEBUG:
        print("Debut de la fonction plot_quadtree")
        
    if isinstance(quadtree, list):
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, edgecolor='black'))
        size /= 2
        if DEBUG:
            print("Affichage de quadtree")
            print(quadtree)

        plot_quadtree(ax, quadtree[0], x, y, size)
        plot_quadtree(ax, quadtree[1], x + size, y, size)
        plot_quadtree(ax, quadtree[2], x, y + size, size)
        plot_quadtree(ax, quadtree[3], x + size, y + size, size)
    else:
        print
        color = quadtree
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=True, color=str(color / 255.0)))
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, edgecolor='red'))


image_path = "cropped_occupancy_grid_image.png"
binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(binary_image, 128, 255, cv2.THRESH_BINARY)

quadtree_graph = build_quadtree_graph(binary_image)
quadtree = build_quadtree(binary_image)


# Affichage graphique
fig, ax = plt.subplots()
ax.set_xlim(0, binary_image.shape[1])
ax.set_ylim(0, binary_image.shape[0])
plot_quadtree(ax, quadtree, 0, 0, binary_image.shape[1])

# Affichage graphique de la structure de l'arbre avec networkx et pydot
plt.figure(figsize=(10, 6))
plot_quadtree_graph(quadtree_graph)
plt.show()

