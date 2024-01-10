#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import heapq
import numpy as np

DEBUG = False

# Variable globale pour le numéro de label
global_node_counter = 0

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
def build_quadtree_graph(image, x=0, y=0, size=None, graph=None, parent=None, edge_label=None, center=None):
    global global_node_counter  # Utilisation de la variable globale

    if size is None:
        size = image.shape[0]

    # Si le graphe n'a pas été construit alors on le construit.
    if graph is None:
        graph = nx.DiGraph()

    # Si case unie = feuille
    if same_color(image):
        color = image[0, 0]
        graph.add_node(color, center=(x + size // 2, y + size // 2), label=global_node_counter)
        global_node_counter += 1  # Incrémentation du compteur global
        # S'il y a un parent on ajoute une arête
        if parent is not None:
            graph.add_edge(parent, global_node_counter -1, label=edge_label)

    # Si besoin de diviser la case
    else:
        node_label = global_node_counter
        graph.add_node(node_label, label=global_node_counter)
        global_node_counter += 1  # Incrémentation du compteur global
        if parent is not None:
            graph.add_edge(parent, node_label, label=edge_label)

        subimages = split(image)
        build_quadtree_graph(subimages[0], x, y, size, graph, node_label, 0)
        build_quadtree_graph(subimages[1], x + size, y, size, graph, node_label, 1)
        build_quadtree_graph(subimages[2], x, y + size, size, graph, node_label, 2)
        build_quadtree_graph(subimages[3], x + size, y + size, size, graph, node_label, 3)

    return graph

# Fonction récursive pour construire le quadtree sur l'image d'origine
# @image une image de taille carré
def build_quadtree(image, x=0, y=0, size=None):
    if size is None:
        size = image.shape[0]

    if same_color(image):
        color = image[0, 0]
        center = (x + size // 2, y + size // 2)
        return color, center
    else:
        subimages = split(image)
        size //= 2
        return [
            build_quadtree(subimages[0], x, y, size),
            build_quadtree(subimages[1], x + size, y, size),
            build_quadtree(subimages[2], x, y + size, size),
            build_quadtree(subimages[3], x + size, y + size, size)
        ]


def dijkstra(graph, start, end):
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start] = 0
    queue = [(0, start)]
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_node == end:
            path = []
            while current_node is not None:
                path.append(graph.nodes[current_node]['center'])
                current_node = graph.pred[current_node]
            return path[::-1]
        
        if current_distance > distances[current_node]:
            continue
        
        for neighbor, data in graph[current_node].items():
            distance = current_distance + data.get('weight', 1)
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
                graph.nodes[neighbor]['pred'] = current_node
                
    return None

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
        a, b= quadtree
        color = a
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=True, color=str(color / 255.0)))
        ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, edgecolor='red'))




image_path = "map.png"
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
#plt.figure(figsize=(10, 6))
print(quadtree_graph)
#plot_quadtree_graph(quadtree_graph)

plt.show()

# Coordonnées du centre du nœud de départ
start_node_center = quadtree[1][0][1]


""" # Coordonnées du centre du nœud d'arrivée
end_node_center = quadtree[2][1][0][1][0][1]

print("Start ", start_node_center)
print("End ", end_node_center)

# Utiliser Dijkstra avec la distance euclidienne
shortest_path = dijkstra(quadtree_graph, start_node_center, end_node_center)

plt.figure(figsize=(10, 6))
# Afficher le chemin sur l'image
for i in range(len(shortest_path)-1):
    x1, y1 = shortest_path[i]
    x2, y2 = shortest_path[i+1]
    plt.plot([y1, y2], [x1, x2], 'r-', linewidth=2)

plt.show() """

