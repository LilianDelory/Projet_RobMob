#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker , MarkerArray
from geometry_msgs.msg import Point
import cv2
import numpy as np
import heapq
import time

DEBUG = False
DEBUG_ZONE_DEPART = True

rviz_subscribed = False  # Drapeau pour indiquer si RViz est abonné ou non

# Charger l'image en niveaux de gris
image = cv2.imread('map_test.png', cv2.IMREAD_GRAYSCALE)

image = cv2.flip(image,0)


# Définir la taille de la fenêtre pour la segmentation
pas = 4
hauteur, largeur = image.shape


# Définir la résolution et l'origine de la grille d'occupation
resolution = 0.05 # m/cell
origin = (-100.0, -100.0) # m

# Convertir les coordonnées de la grille d'occupation en coordonnées du monde
def convert_center_to_world(zone):
    x = (zone['center'][0]/2 + 1800) * resolution + origin[0]
    y = (- zone['center'][1]/2 + 2200) * resolution + origin[1]
    return (x, y) 


def afficher_positions_rviz(chemin):
    rospy.init_node("marqueur_publisher")
    print("Initialisation du noeud...")

    marker_array = MarkerArray()
    marker_pub = rospy.Publisher("/Publication", MarkerArray, queue_size=1)

    # Attendre que le nœud ROS soit correctement initialisé
    rospy.sleep(1)
    i = 0

    while not rospy.is_shutdown():
        if marker_pub.get_num_connections() == 0:
            rospy.loginfo("En attente d'abonnement de RViz...")
            rospy.sleep(1)
            continue

        for position in chemin:
            x, y = position

            marker = Marker()
            marker.header.frame_id = "map"  
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.id = i 
            marker.pose.position.z = 0.5 
            point = Point()
            point.x = x
            point.y = y
            marker.points.append(point)
            marker_array.markers.append(marker)
            i += 1

        # Publier le marqueur une seule fois
        marker_pub.publish(marker_array)
        print("Publication du marqueur...")

        # Rate de publication
        rate = rospy.Rate(1)  # Fréquence de publication (1 Hz dans cet exemple)
        rate.sleep()

        break  # Sortir de la boucle après une itération """




""" # Afficher les positions dans RViz
def afficher_positions_rviz(chemin):

    # Initialiser le nœud ROS
    rospy.init_node("marqueur_publisher")
    print("Initialisation du noeud...")

    marker_array = MarkerArray()

    # Créer un éditeur de messages Marker
    marker_pub = rospy.Publisher("/Publication", MarkerArray, queue_size=1)

     # S'abonner au topic sur lequel RViz s'abonne normalement
    rospy.Subscriber("/Publication", MarkerArray, rviz_subscribe_callback)

    # Attendre que le nœud ROS soit correctement initialisé
    rospy.sleep(1)
    i = 0
    # Ajouter les points aux coordonnées spécifiées
    for position in chemin:
        x, y = position

        # Créer un éditeur de messages Marker
        marker = Marker()
        marker.header.frame_id = "map"  
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.scale.x = 1
        marker.scale.y = 10000.2
        marker.scale.z = 1000.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.id = i 

        marker.pose.position.z = 0.5  
        marker_array.markers.append(marker)
        i += 1

    # Publier le marqueur une seule fois
    # La boucle principale
    while not rospy.is_shutdown():
        # Attendre que RViz s'abonne
        if not rviz_subscribed:
            rospy.loginfo("En attente d'abonnement de RViz...")
            rospy.sleep(1)
            continue

        # Publier le marqueur une seule fois
        marker_pub.publish(marker_array)
        print("Publication du marqueur...")
        time.sleep(5)
        # Rate de publication
        rate = rospy.Rate(1)  # Fréquence de publication (1 Hz dans cet exemple)
        rate.sleep() """


# Parcourir l'image avec la fenêtre spécifiée et identifier les zones robot et obstacle
zones_robot = []        # Zones vertes   
zones_obstacle = []     # Zones rouges
zones_globales = []     # Toutes les zones

# Compteur de Case : i
i = 1 
for y in range(0, hauteur, pas):
    for x in range(0, largeur, pas):
        fenetre = image[y:y + pas, x:x + pas]
        x_center = x + pas // 2
        y_center = y + pas // 2
        
        # Vérifier si tous les pixels de la fenêtre sont noirs
        if np.all(fenetre == 0):
            zones_robot.append({'x': x, 'y': y, 'x_end': x + pas, 'y_end': y + pas, 'numero': i, 'Zone': 'robot', 'center': (x_center, y_center)})
            zones_globales.append({'x': x, 'y': y, 'x_end': x + pas, 'y_end': y + pas, 'numero': i, 'Zone': 'robot', 'center': (x_center, y_center)})
        else:
            zones_obstacle.append({'x': x, 'y': y, 'x_end': x + pas, 'y_end': y + pas, 'numero': i, 'Zone': 'obstacle', 'center': (x_center, y_center)})
            zones_globales.append({'x': x, 'y': y, 'x_end': x + pas, 'y_end': y + pas, 'numero': i, 'Zone': 'obstacle' , 'center': (x_center, y_center)})

        
        i += 1

# Dessiner les zones sur l'image originale (à titre d'exemple)
image_colore = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for zone in zones_robot:
    #print(zone)
    cv2.rectangle(image_colore, (zone['x'], zone['y']), (zone['x_end'], zone['y_end']), (255, 255, 255), pas)

for zone in zones_obstacle:
    #print(zone)
    cv2.rectangle(image_colore, (zone['x'], zone['y']), (zone['x_end'], zone['y_end']), (0, 0, 0), pas)



compteur_robot = 0
############ Trouver les zones adjacentes = VOISINS ############
for zone in zones_globales:
    if zone['Zone'] == 'robot':
        zone['voisins'] = [zone['numero']-1,zone['numero']+1,zone['numero']-largeur//pas,zone['numero']+largeur//pas, zone['numero']-largeur//pas +1,zone['numero']-largeur//pas -1 ,zone['numero']+largeur//pas +1 ,zone['numero']+largeur//pas -1 ]
        
        #Tester si les voisins sont des obstacles
        for voisin in zone['voisins']:
            if zones_globales[voisin-1]['Zone'] == 'obstacle':
                zone['voisins'].remove(voisin)
        #print(zone['voisins'])

        #Ajouter les voisins à la liste des zones robot
        zones_robot[compteur_robot]['voisins'] = zone['voisins']
        compteur_robot += 1


def distance_zone(zones_globales,zone1, zone2):
    # Calculer la distance entre deux zones
    x1, y1 = zones_globales[zone1-1]['center']
    x2, y2 = zones_globales[zone2-1]['center']
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


################## DIJKSTRA #####################
def dijkstra(zones_robot, zones_globales, start, goal):
    # Implémentation de l'algorithme de Dijkstra
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, node, path) = heapq.heappop(queue)

        if node not in visited:
            visited.add(node)

            if node == goal:
                return path + [node]

            #print("Node : ", node)
            for neighbor in zones_globales[node-1]['voisins']:
                if(zones_globales[neighbor-1]['Zone'] == 'obstacle'):
                    continue
                else:
                    #print('neighbor: ',neighbor)
                    heapq.heappush(queue, (cost + distance_zone(zones_globales,node,neighbor), neighbor, path + [node]))

    return None

# Points de départ et d'arrivée
num = np.random.randint(1, len(zones_robot)) 
point_depart = zones_robot[num]['numero']
point_depart = zones_globales[point_depart-1]['numero']

#point_depart = 24641




num = np.random.randint(1, len(zones_robot)) 
point_arrivee = zones_robot[num]['numero']
point_arrivee = zones_globales[point_arrivee-1]['numero']

#point_arrivee = 19288


if DEBUG_ZONE_DEPART:
    print(zones_globales[point_depart-1])
    print(zones_globales[point_arrivee-1])


# Utiliser Dijkstra pour trouver le chemin le plus court
chemin_plus_court = dijkstra(zones_robot,zones_globales, point_depart, point_arrivee)

if DEBUG:
    print(chemin_plus_court)

List_position = []
for i in chemin_plus_court:
    List_position.append(convert_center_to_world(zones_globales[i-1]))


if DEBUG:
    print("\nListe_positions :",List_position,'\n')


# Dessiner le chemin sur l'image colore
if chemin_plus_court:
    for point in chemin_plus_court:
        zone = zones_globales[point - 1]
        cv2.rectangle(image_colore, (zone['x'], zone['y']), (zone['x_end'], zone['y_end']), (0, 0, 255), -1)
else:
    print("Le chemin le plus court n'a pas pu être trouvé.")

# Afficher les positions dans RViz
print("Affichage des positions dans RViz...")
afficher_positions_rviz(List_position)



#Aplliquer une couleur différente à la première et dernière zone
cv2.circle(image_colore, zones_globales[point_depart-1]['center'], 5, (0, 255, 255), -1)
cv2.circle(image_colore, zones_globales[point_arrivee-1]['center'], 5, (255, 0, 255), -1)
 

""" # Afficher l'image résultante
cv2.imshow('Image segmentee', image_colore)
cv2.waitKey(0)
cv2.destroyAllWindows()

 """

