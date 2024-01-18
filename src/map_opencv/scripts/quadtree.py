#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import heapq
import numpy as np
import time

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
        print(size)

    # Si le graphe n'a pas été construit alors on le construit.
    if graph is None:
        graph = nx.DiGraph()



    # Si case unie = feuille
    if same_color(image):
        color = image[0, 0]
        graph.add_node(global_node_counter, couleur=color, center=(x + size // 2, y + size // 2), label=global_node_counter, parent=parent, coin=edge_label, feuille = True, taille = size)
        global_node_counter += 1
        #S'il y a un parent on ajoute une arête
        if parent is not None:
            graph.add_edge(parent, global_node_counter -1, label=edge_label)
            
    
    # Si besoin de diviser la case
    else:
        node_label = global_node_counter
        graph.add_node(node_label, label=global_node_counter, parent=parent, coin = edge_label, feuille = False, taille = size)
        global_node_counter += 1  # Incrémentation du compteur global
        if parent is not None:
            graph.add_edge(parent, node_label, label=edge_label)

        subimages = split(image)
        size //= 2
        build_quadtree_graph(subimages[0], x, y, size, graph, node_label, 0)
        build_quadtree_graph(subimages[1], x + size, y, size, graph, node_label, 1)
        build_quadtree_graph(subimages[2], x, y + size, size, graph, node_label, 2)
        build_quadtree_graph(subimages[3], x + size, y + size, size, graph, node_label, 3)

    return graph



def find_predecessor(graph,feuille_etudie):
    find_pred = []
    while(graph.nodes[feuille_etudie]['parent'] is not None):
        find_pred.append(feuille_etudie)
        feuille_etudie = graph.nodes[feuille_etudie]['parent']

    return find_pred




def near_neighbours(graph, feuille_etudie, carre_etudie):
    parent = graph.nodes[feuille_etudie]['parent']
    children = list(graph.successors(parent))
    neigh = []

    # Utilisation de next avec une compréhension de liste pour trouver le premier nœud qui correspond à la condition
    neighbour_node = next((child for child in children if graph.nodes[child]['coin'] == carre_etudie), None)

    if neighbour_node is not None and not graph.nodes[neighbour_node]['feuille']:
        # Si le nœud trouvé n'est pas une feuille, recherche récursive des voisins dans les enfants non-feuilles
        neigh.extend(find_leaf_neighbours(graph, neighbour_node, carre_etudie, graph.nodes[feuille_etudie]['coin']))
        return neigh
    else:
        neigh.append(neighbour_node)
        return neigh

def find_leaf_neighbours(graph, parent_node, carre_etudie, position_feuille_origine):
    # Initialisation d'une liste pour stocker les voisins trouvés
    leaf_neighbours = []

    # Récupération des enfants du parent_node
    children = list(graph.successors(parent_node))

    # Détermination des enfants spécifiques à considérer en fonction de la position de la feuille étudiée
    relevant_children = get_relevant_children(carre_etudie, position_feuille_origine)

    # Parcours des enfants pour trouver les voisins
    for child in children:
        if graph.nodes[child]['coin'] in relevant_children:
            if not graph.nodes[child]['feuille']:
                # Si le nœud n'est pas une feuille, recherche récursive dans ses enfants non-feuilles
                leaf_neighbours.extend(find_leaf_neighbours(graph, child, carre_etudie, position_feuille_origine))
            else:
                # Si le nœud est une feuille, ajouter à la liste des voisins si du côté correspondant
                leaf_neighbours.append(child)

    return leaf_neighbours

def get_relevant_children(carre_etudie, position_feuille_origine):
    # Fonction pour obtenir la liste des enfants spécifiques à considérer en fonction de la position de la feuille étudiée
    if(position_feuille_origine == 0):
        if carre_etudie == 1:
            return [0, 2]
        
        elif carre_etudie == 2:
            return [0, 1]
        
    if(position_feuille_origine == 1):
        if carre_etudie == 0:
            return [1, 3]
        
        elif carre_etudie == 3:
            return [0, 1]
    
    if(position_feuille_origine == 3):
        if carre_etudie == 1:
            return [2, 3]
        
        elif carre_etudie == 2:
            return [3, 1]
    
    if(position_feuille_origine == 2):
        if carre_etudie == 3:
            return [2, 0]
        
        elif carre_etudie == 0:
            return [2, 3]
    else:
        return []  # Valeur par défaut si carre_etudie n'est pas dans la plage attendue

        
def is_neighbour(coin_position, carre_etudie):
    # Fonction pour vérifier si la feuille à la position 'coin_position' est voisine de la feuille 'carre_etudie'
    # Vous pouvez définir votre propre logique ici en fonction de la numérotation des coins que vous avez décrite
    # Dans cet exemple, la logique est basée sur la numérotation spécifique que vous avez fournie (0, 1, 2, 3)
    if carre_etudie == 0:
        return coin_position in [1, 2]
    elif carre_etudie == 1:
        return coin_position in [0, 3]
    elif carre_etudie == 2:
        return coin_position in [0, 3]
    elif carre_etudie == 3:
        return coin_position in [1, 2]
    else:
        return False  # Valeur par défaut, si carre_etudie n'est pas dans la plage attendue



# Trouve les voisins qui ne sont pas directement issu du parent direct
def find_neighbours_cousin(graph, feuille_etudie, feuille_origine = None, generation_count=0):
    parent = graph.nodes[feuille_etudie]['parent']
    
    print("Compteur : " + str(generation_count))

    neigh = []

    if feuille_origine is None:
        feuille_origine = feuille_etudie

    position_feuille_origine = graph.nodes[feuille_origine]['coin']

    if parent is not None:
        
        parent_coin = graph.nodes[parent]['coin']
        #print("Parent Coin " + str(parent_coin))
        if(parent_coin == None):
            print("Retour")
            return(neigh)

        # Si le coin du parent est identique à celui de la feuille étudiée, continuer de remonter
        if parent_coin == graph.nodes[feuille_etudie]['coin']:
            print("Montée de Parent")
            neigh.extend(find_neighbours_cousin(graph, parent, feuille_origine, generation_count + 1))
            return neigh
        
    # S'il y a une différence de coin ou si on a atteint la racine de l'arbre, on peut chercher parmi les voisins potentiels
    neigh.extend(find_leaf_neighbours_recursive(graph, parent, feuille_origine, generation_count+1))

    return neigh


def get_relevant_children_cousin(position_feuille_origine, coin_parent):
    if(position_feuille_origine == 0):
        if(coin_parent == 0 or coin_parent==3):
            return [(1, 'haut'), (2,'droite')]
        
        if(coin_parent == 2):
            return[(0,'haut')]
        
        if(coin_parent == 1):
            return[(0,'droite')]
        
    if(position_feuille_origine == 1):
        if(coin_parent == 3):
            return [(1,'haut')]
        
        if(coin_parent == 0):
            return [(1,'gauche')]
        
        if(coin_parent == 2):
            return [(0,'haut'), (3,'gauche')]
        
    if(position_feuille_origine == 2):
        if(coin_parent == 0):
            return [(2,'bas')]
        
        if(coin_parent == 3):
            return[(2,'droite')]
        
        if(coin_parent == 1):
            return [(0,'droite'), (3,'bas')]
        
    if(position_feuille_origine == 3):
        if(coin_parent == 0 or coin_parent == 3):
            return [(2,'bas'),(1,'gauche')]
        
        if(coin_parent == 2):
            return [(3,'gauche')]
        
        if (coin_parent == 1):
            return [(3,'bas')]
        
    else:
        return []  # Valeur par défaut 
    

def find_leaf_from_cousin(graph, cousin_node, feuille_origine, direction, generation_count):
    position_feuille_origine = graph.nodes[feuille_origine]['coin']
    print("Compt " + str(generation_count))
    # Initialisation d'une liste pour stocker les voisins trouvés
    leaf_neighbours = []

    # Détermination des enfants spécifiques à considérer en fonction de la position de la feuille étudiée
    if(direction == 'bas'):
        if(position_feuille_origine == 2):
            relevant_children = [0, 1]
        
        if(position_feuille_origine == 3):
            relevant_children = [1, 0]
    
    if(direction == 'haut'):
        if(position_feuille_origine == 0):
            relevant_children = [2, 3]
        
        if(position_feuille_origine == 1):
            relevant_children = [3, 2]

    if(direction == 'droite'):
        if(position_feuille_origine == 0):
            relevant_children = [1,3]
        
        if(position_feuille_origine == 2):
            relevant_children = [3,1]

    if(direction == 'gauche'):
        if(position_feuille_origine == 1):
            relevant_children = [0,3]
        
        if(position_feuille_origine == 3):
            relevant_children = [3,0]
            
    # Récupération des enfants du cousin
    children = list(graph.successors(cousin_node))

    # Parcours des enfants pour trouver les voisins
    for child in children:
        if generation_count <= 0 :
            #time.sleep(5)
            if graph.nodes[child]['coin'] in relevant_children:
                if not graph.nodes[child]['feuille']:
                    # Si le nœud n'est pas une feuille, recherche récursive dans ses enfants non-feuilles
                    leaf_neighbours.extend(find_leaf_from_cousin(graph, child, feuille_origine, direction, generation_count - 1))
                else:
                    # Si le nœud est une feuille, ajouter à la liste des voisins si du côté correspondant
                    leaf_neighbours.append(child)
        else:
            
            if graph.nodes[child]['coin'] == relevant_children[0]:
                if not graph.nodes[child]['feuille']:
                    # Si le nœud n'est pas une feuille, recherche récursive dans ses enfants non-feuilles
                    leaf_neighbours.extend(find_leaf_from_cousin(graph, child, feuille_origine, direction, generation_count - 1))
                else:
                    # Si le nœud est une feuille, ajouter à la liste des voisins si du côté correspondant
                    leaf_neighbours.append(child)


    return leaf_neighbours

def is_side_last_leaf_different(position_prec, position_actu):

    if(position_actu == 0 or position_prec == 2):
        if(position_prec == 0 or position_prec == 2):
            return False
        else:
            return True
        
    if(position_actu == 3 or position_prec == 1):
        if(position_prec == 3 or position_prec == 1):
            return False
        else:
            return True
        

def euclidean_distance(node1, node2, graph):
    x1, y1 = graph.nodes[node1]['center'][0], graph.nodes[node1]['center'][1]
    x2, y2 = graph.nodes[node2]['center'][0], graph.nodes[node2]['center'][1]

    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return distance

def find_last_leaf_neighbours(graph, square, feuille_origine):
    leaves = is_leaf(graph, square)

    if leaves is None:
        return []

    # Initialiser la distance minimale comme l'infini
    min_distance = float('inf')
    
    # Liste pour stocker les feuilles les plus proches
    closest_leaves = []

    # Calculer la distance euclidienne entre la feuille d'origine et toutes les feuilles
    for leaf in leaves:
        distance = euclidean_distance(feuille_origine, leaf, graph)
        
        # Mettre à jour la liste des feuilles les plus proches si une feuille plus proche est trouvée
        if distance < min_distance:
            min_distance = distance
            closest_leaves = [leaf]
        elif distance == min_distance:
            # Si deux feuilles sont à la même distance, les ajouter toutes les deux
            closest_leaves.append(leaf)

    return closest_leaves


def is_leaf(graph, node):
    # Vérifier si le nœud est une feuille
    if graph.nodes[node]['feuille']:
        return node
    else:
        # Si le nœud n'est pas une feuille, explorer récursivement ses enfants
        children = list(graph.successors(node))
        for child in children:
            result = is_leaf(graph, child)
            if result is not None:
                return result  # Retourner le nœud feuille trouvé
            
            return None  # Aucun nœud feuille trouvé dans les enfants, retourner None

    

""" def find_neighbours_last_square(graph, parent, feuille_origine, coin):

    position_feuille_origine = graph.nodes[feuille_origine]['coin']
    parent_coin = graph.nodes[parent]['coin']
    if(position_feuille_origine == 2):
            if(parent_coin == 0):
                position_feuille_etudie = 2
            if(parent_coin == 1):
                position_feuille_etudie = 3

            if(parent_coin == 2):
                position_feuille_etudie = 2
            if(parent_coin == 3):
                position_feuille_etudie = 3

    if(parent_coin == 0):
        if(coin == 0):

    
    # Récupération des frères de mon parent
    children = list(graph.successors(parent))
    for child in children:
        if(graph.nodes[child]['coin']==position_carre_etudie):
            return find_last_leaf_neighbours(graph, child, feuille_origine)
             """
    

def find_leaf_neighbours_recursive(graph, parent_node, feuille_origine, generation_count):
    # Initialisation d'une liste pour stocker les voisins trouvés
    leaf_neighbours = []
    print("Compt " + str(generation_count))

    position_feuille_origine = graph.nodes[feuille_origine]['coin']
   
    parent = graph.nodes[parent_node]['parent']

    if(parent is None):
        print("Retour Origine = Coin")
        return leaf_neighbours

    # Récupération des frères de mon parent
    children = list(graph.successors(parent))

    # Détermination des enfants spécifiques à considérer en fonction de la position de la feuille étudiée
    relevant_children = get_relevant_children_cousin(position_feuille_origine, graph.nodes[parent_node]['coin'])


    #print("Liste de relevant children " + str(relevant_children))
    #print("Liste des enfants : " + str(children))

    if (len(relevant_children) == 2):
        #time.sleep(5)
        for child in children:
            for node, direction in [(node, direction) for node, direction in relevant_children]:
                    if graph.nodes[child]['coin'] == node:
                        if not graph.nodes[child]['feuille']:
                            # Si le nœud n'est pas une feuille, recherche récursive dans ses enfants non-feuilles
                            leaf_neighbours.extend(find_leaf_from_cousin(graph, child, feuille_origine, direction, generation_count))
                        else:
                            # Si le nœud est une feuille, ajouter à la liste des voisins avec le compteur de générations
                            leaf_neighbours.append(child)
    
    if(len(relevant_children)==1):
        print("Tzilllz")
        for child in children:
                if graph.nodes[child]['coin'] == relevant_children[0][0]:
                    print
                    if not graph.nodes[child]['feuille']:
                        # Si le nœud n'est pas une feuille, recherche récursive dans ses enfants non-feuilles
                        leaf_neighbours.extend(find_leaf_from_cousin(graph, child, feuille_origine, relevant_children[0][1], generation_count))
                    else:
                        # Si le nœud est une feuille, ajouter à la liste des voisins avec le compteur de générations
                        leaf_neighbours.append(child)
        
        """  time.sleep(5)
        #parent = graph.nodes[parent]['parent']

        if parent is not None :
            leaf_neighbours.extend(find_neighbours_last_square(graph,parent,feuille_origine, relevant_children[0][0])) """


    return leaf_neighbours





def info_nodes(graph):
    print(graph.__len__())
    for node in graph:
        print("---------- Info du noeud")
        print(graph.nodes[node])
        neigh = []
        if graph.nodes[node]['feuille']:
            graph.nodes[node]['prec'] = find_predecessor(graph, node)
            if(graph.nodes[node]['coin'] == 0):
                neigh.extend(near_neighbours(graph,node,1))
                neigh.extend(near_neighbours(graph,node,2))
            if(graph.nodes[node]['coin'] == 1):
                neigh.extend(near_neighbours(graph,node,0))
                neigh.extend(near_neighbours(graph,node,3))
            if(graph.nodes[node]['coin'] == 2):
                neigh.extend(near_neighbours(graph,node,0))
                neigh.extend(near_neighbours(graph,node,3))
            if(graph.nodes[node]['coin'] == 3):
                neigh.extend(near_neighbours(graph,node,1))
                neigh.extend(near_neighbours(graph,node,2))

            graph.nodes[node]['voisins'] = neigh
            print("Algo")
            print(graph.nodes[node]['voisins'])
            print("Position des voisins")
            for nei in neigh:
                print(graph.nodes[nei]['center'])


def info_nodes2(graph):
    print(graph.__len__())
    for node in graph:
        graph.nodes[node]['prec'] = find_predecessor(graph, node)
        print("---------- Info du noeud")
        print(graph.nodes[node])
        neigh = []
        if graph.nodes[node]['feuille']:
            neigh = find_neighbours_cousin(graph, node, generation_count=0)
            v = graph.nodes[node]['voisins']
            v.extend(neigh)
            graph.nodes[node]['voisins'] = v
            print("Algo")
            print(neigh)
            if(len(neigh) != 0):
                for nei in neigh:
                    print(graph.nodes[nei]['center'])


def plot_neighbors(ax, graph, feuille_etudiee):
    neighbors = graph.nodes[feuille_etudiee]['voisins']
    print("Centre " + str(graph.nodes[feuille_etudiee]['center']))
    
    if neighbors:
        for neighbor in neighbors:
            neighbor_center = graph.nodes[neighbor]['center']
            color = 'blue'  # Couleur bleue pour les feuilles avec des voisins

            center = graph.nodes[neighbor]['center']
            size = graph.nodes[neighbor]['taille']
            ax.add_patch(plt.Rectangle((center[0] - size / 2, center[1] - size / 2), size, size, fill=True, color=color))
            ax.add_patch(plt.Rectangle((center[0] - size / 2, center[1] - size / 2), size, size, fill=False, edgecolor='black'))
    




# Fonction récursive pour construire le quadtree sur l'image d'origine
# @image une image de taille carré
def build_quadtree(image, x=0, y=0, size=None):
    if size is None:
        size = image.shape[0]

    if same_color(image):
        color = image[0, 0]
        #print('Size ' + str(size))

        if(size == 1):
            center = (x + 0.5, y + 0.5)
        else:
            center = (x + size / 2, y + size / 2)
        return color, center
    else:
        subimages = split(image)
        size /= 2
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

quadtree_graph = build_quadtree_graph(binary_image,size=512)

info_nodes(quadtree_graph)
info_nodes2(quadtree_graph)




""" plot_quadtree_graph(quadtree_graph)
plt.show() """

quadtree = build_quadtree(binary_image)
# Affichage graphique
fig, ax = plt.subplots()
ax.set_xlim(0, binary_image.shape[1])
ax.set_ylim(0, binary_image.shape[0])
plot_quadtree(ax, quadtree, 0, 0, binary_image.shape[1])
# Changer le pas des axes X
custom_xticks = np.arange(0, 512, 16)  # Définir les positions des ticks personnalisées
plt.xticks(custom_xticks)

plt.yticks(custom_xticks)

# Tracer les voisins de la feuille spécifiée
plot_neighbors(ax, quadtree_graph, 4791)

#plot_neighbors(ax, quadtree_graph, 24)

#plot_neighbors(ax, quadtree_graph, 102)

plt.show()


"""
# Affichage graphique de la structure de l'arbre avec networkx et pydot
#plt.figure(figsize=(10, 6))
print(quadtree_graph)

"""

""""
# Coordonnées du centre du nœud de départ
start_node_center = quadtree[1][0][1]


# Coordonnées du centre du nœud d'arrivée
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

