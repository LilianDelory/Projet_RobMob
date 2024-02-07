#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import cv2
import numpy as np
import heapq
import time
import tf
from compileall import compile_path
from os import path
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool



DEBUG = False
DEBUG_ZONE_DEPART = True

mode_rviz = True  # Drapeau pour indiquer si RViz est utilisé ou non


# Variable globale pour stocker les points du chemin
path_points = []

#Compteur Global
compteur = 1

#Seuile de précision pour l'erreur de position
seuil = 0.3

#Position globale
pose  = [0,0,0]

#Gains pour loi de commande
K1 = 0.8
K2 = 0.3

#Paramètres du robot
L1 = 0.15


# Charger l'image en niveaux de gris
# Ajout d'une vérification de l'existence du fichier

try:
    image = cv2.imread('/home/lilian/new_catkin_ws/my_occupancy_grid.png', cv2.IMREAD_GRAYSCALE)
    print ("Lecture de l'image...")
except FileNotFoundError:
    print("Le fichier occupancy_grid_image.png n'existe pas dans le dossier home/new_catkin_ws")
    

image = cv2.flip(image,0)

#Lecture d'un fichier.yaml pour obtenir la résolution et l'origine avec vérification de l'existence du fichier
# Le fichier Yaml se trouve dans le dossier home/new_catkin_ws

try:   
    with open('/home/lilian/new_catkin_ws/my_occupancy_grid.yaml') as f:
        print("Lecture du fichier yaml...")
        lines = f.readlines()
        for line in lines:

            if 'resolution' in line:
                resolution = float(line.split()[1])
                print("Résolution : ", resolution)

            if 'origin' in line:
                origin_1 = line.split()[1]
                origin_2 = line.split()[2]

                if DEBUG:
                    print ("Origine : ", origin_1)
                    print ("Origine : ", origin_2)

                origin_1_cleaned = origin_1.replace('[', '').replace(']', '').replace(',', '')
                origin_2_cleaned = origin_2.replace('[', '').replace(']', '').replace(',', '')
                
                if DEBUG:
                    print("Origine 1 nettoyée : ", origin_1_cleaned)
                    print("Origine 2 nettoyée : ", origin_2_cleaned)

                origin = (float(origin_1_cleaned), float(origin_2_cleaned))
                print("Origine : ", origin)

except FileNotFoundError:
    print("Le fichier my_occupancy_grid.yaml n'existe pas dans le dossier home/new_catkin_ws")

#Donner le type de 'image'
    if DEBUG:
        print("Type de l'image : ", type(image))


# Définir la taille de la fenêtre pour la segmentation
pas = 8
hauteur, largeur = image.shape

# Lecture du pgm pour obtenir sa taille
# Ajout d'une vérification de l'existence du fichier
# Le fichier se trouve en home/new_catkin_ws 

try:
    with open('/home/lilian/new_catkin_ws/my_occupancy_grid.pgm', 'rb') as f:
        print("Lecture du fichier pgm...")
        f.readline()  # P5
        f.readline()  # Commentaire
        largeur_pgm, hauteur_pgm = [int(i) for i in f.readline().split()]
        print ("Largeur :", largeur_pgm, "Hauteur :", hauteur_pgm)
        f.readline()  # 255
except FileNotFoundError:
    print("Le fichier my_occupancy_grid.pgm n'existe pas dans le dossier home/new_catkin_ws")


iamge_cropped = False
#Détermine si l'image est cropped 
if (hauteur_pgm != hauteur) or (largeur_pgm != largeur):
    print("L'image est cropped")
    iamge_cropped = True

# Position du robot
position_globale = [0,0]



# Convertir les coordonnées de la grille d'occupation en coordonnées du monde
def convert_center_to_world(zone):

    ############ CAS 1 : IMAGE CROPPED -> 800x800 ############
    if iamge_cropped:
        x = (zone['center'][0] + largeur_pgm/2 - 400) * resolution + origin[0]
        y = (- zone['center'][1] + hauteur_pgm/2 + 400 ) * resolution + origin[1]
    
    ############ CAS 2 : IMAGE NON CROPPED ############
    else :
        x = (zone['center'][0]) * resolution + origin[0]
        y = (hauteur - zone['center'][1]) * resolution + origin[1]
        
    return (x, y)
    

# Convertir les coordonnées du monde en coordonnées de la grille d'occupation
def convert_world_to_center(position):

    ############ CAS 1 : IMAGE CROPPED -> 800x800 ############
    if iamge_cropped:
        x = (position[0] - origin[0]) / resolution + 400 - largeur_pgm/2
        y = (- position[1] + origin[1]) / resolution + 400 + hauteur_pgm/2
        
    ############ CAS 2 : IMAGE NON CROPPED ############
    else : 
        x = (position[0] - origin[0]) / resolution
        y = (- position[1] + origin[1]) / resolution + hauteur
        
    return (x, y)


def afficher_positions_rviz(chemin):
    #rospy.init_node("marqueur_publisher")
    print("Initialisation du noeud...")

    marker_array = MarkerArray()
    marker_pub = rospy.Publisher("/Publication", MarkerArray, queue_size=1)

    # Attendre que le nœud ROS soit correctement initialisé
    rospy.sleep(1)
    i = 0

    while not rospy.is_shutdown():
        if marker_pub.get_num_connections() < 1:
            rospy.loginfo("En attente d'abonnement de RViz...")
            print("Nombre de connections : ", marker_pub.get_num_connections() )
            rospy.sleep(0.01)
            continue
        
        
        for position in chemin:
            x, y = position

            marker = Marker()
            marker.header.frame_id = "map"  
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.scale.x = 0.17
            marker.scale.y = 0.17
            marker.scale.z = 0.17
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.id = i 
            marker.pose.position.z = 0.02 
            point = Point()
            point.x = x
            point.y = y
            marker.points.append(point)
            marker_array.markers.append(marker)
            path_points.append([point.x, point.y])
            i += 1

            print("################# POSITION #################", position)
            print("#############s##### POINT ##################", point)

        # Publier le marqueur une seule fois
        marker_pub.publish(marker_array)
        print("Publication du marqueur...")

        # Rate de publication
        rate = rospy.Rate(1)  # Fréquence de publication (1 Hz dans cet exemple)
        rate.sleep()
        break  # Sortir de la boucle après une itération """


# Parcourir l'image avec la fenêtre spécifiée et identifier les zones robot et obstacle
zones_robot = []        # Zones vertes   
zones_obstacle = []     # Zones rouges
zones_globales = []     # Toutes les zones

# Compteur de Case : i
i = 1 
for y in range(0, hauteur, pas):
    for x in range(0, largeur, pas):
        fenetre = image[y:y + pas, x:x + pas]
        x_center = x + (pas // 2)
        y_center = y + (pas // 2)
        
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

            if len(zones_globales[node-1]['voisins']) != 8:
                rajout = 1000
            else:
                rajout = 0


            #print("Node : ", node)
            for neighbor in zones_globales[node-1]['voisins']:
                if(zones_globales[neighbor-1]['Zone'] == 'obstacle'):
                    continue
                else:
                    #print('neighbor: ',neighbor)
                    heapq.heappush(queue, (cost + rajout + distance_zone(zones_globales,node,neighbor), neighbor, path + [node]))

    return None



def zone_proche_robot():
    global position_globale

    #rospy.init_node('tf_listener', anonymous=True)
    #while not rospy.is_shutdown():
    tf_callback()
    #rate.sleep()
        

    print("Position globale : ", position_globale)
    new_pose = convert_world_to_center(position_globale)
    print("Position globale convertie : ", new_pose)

    #Trouver la zone la plus proche du robot
    distance = 10000000
    numero_case = 0

    for zone in zones_robot:
        x,y = zone['center']
        if (np.sqrt((new_pose[0]-x)**2 + (new_pose[1]-y)**2) < distance):
            distance = np.sqrt((new_pose[0]-x)**2 + (new_pose[1]-y)**2)
            numero_case = zone['numero']

    return numero_case



def tf_callback():
    global position_globale
    listener = tf.TransformListener()

    try:
        listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(10))
        (trans, rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        position_globale = [trans[0], trans[1]]
        print("Position globale dans tf_callback : ", position_globale)
    except tf.Exception as e:
        rospy.logwarn("Failed to get the transformation: %s", str(e))




def trouver_base(zones_robot):
    #Trouver la zone la plus proche du robot

    position_base = [-9.3, -2.3]
    new_pose = convert_world_to_center(position_base)

    #Trouver la zone la plus proche du robot
    distance = 10000000
    numero_case = 0

    for zone in zones_robot:
        x,y = zone['center']
        if (np.sqrt((new_pose[0]-x)**2 + (new_pose[1]-y)**2) < distance):
            distance = np.sqrt((new_pose[0]-x)**2 + (new_pose[1]-y)**2)
            numero_case = zone['numero']

    return numero_case



    distance = 10000000
    numero_case = 0
    for zone in zones_robot:
        x,y = zone['center']
        if (np.sqrt((largeur/2-x)**2 + (hauteur/2-y)**2) < distance):
            distance = np.sqrt((largeur/2-x)**2 + (hauteur/2-y)**2)
            numero_case = zone['numero']

    return numero_case


def odom_callback():
    global pose
    listener = tf.TransformListener()
    print("test2")

    try:
        listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(1))
        (trans, rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        pose = [trans[0], trans[1], euler_from_quaternion(rot)[2]]
    except tf.Exception as e:
        rospy.logwarn("Failed to get the transformation: %s", str(e))



def commande(pose, path_points):
    global compteur
    global K1
    global K2
    global L1
    
    print("###### POSE ###### :",pose)
    
    theta = pose[2]
    xp = pose[0] + L1*np.cos(theta)
    yp = pose[1] + L1*np.sin(theta)

    xr = path_points[compteur][0]
    yr = path_points[compteur][1]

    erreur  = np.sqrt((xr - xp)**2 + (yr - yp)**2 )

    print('############# XP ###############:', xp)
    print('############# XR ###############:', xr)
    print('############# YP ###############:', yp)
    print('############# YR ###############:', yr)
    print('########### ERREUR #############:', erreur)


    if erreur < seuil : 
        compteur += 1

    if(compteur == len(path_points)):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0
        cmd_vel_msg.angular.z = 0
        cmd_vel_pub.publish(cmd_vel_msg)
        print(cmd_vel_msg)
        

    
    else:
        print("compteur :", compteur)
        v1 = -K1*(xp - xr)
        v2 = -K2*(yp - yr)
        v = [[v1],[v2]]


        rot = [[np.cos(theta), -L1*np.sin(theta)],[np.sin(theta), L1*np.cos(theta)]]
        inv_rot = np.linalg.inv(rot)
        u = np.dot(inv_rot,v)


        u1 = u[0][0]*3.8
        u2 = u[1][0]*0.5



        print('################ U1 ################ :', u1)
        print('################ U2 ################ :', u2)

        # Créer et publier la commande Twist
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = u1
        cmd_vel_msg.angular.z = u2
        cmd_vel_pub.publish(cmd_vel_msg)
        print(cmd_vel_msg)


def listener():
    #rospy.init_node('odom_listener', anonymous=True)
    print("test2")
    rate = rospy.Rate(10)
    print(compteur)
    while not rospy.is_shutdown() and compteur < len(path_points):
        print("test3")
        odom_callback()
        commande(pose, path_points)
        rate.sleep()

    rospy.spin()



def button_callback(msg):
    if msg.data:
        rospy.loginfo("Le bouton A a été enfoncé.")

        ######### POINTS DE DEPART ET D'ARRIVEE #########

        if mode_rviz:
            point_depart = zone_proche_robot()
            point_arrivee = trouver_base(zones_robot)

            #point_depart = 35103
            #point_arrivee = 57056

        else : 

            point_depart = zone_proche_robot()
            point_arrivee = trouver_base(zones_robot)

            #point_depart = 3538
            #point_arrivee = 4434


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
        if mode_rviz:
            afficher_positions_rviz(List_position)
            print(List_position)
            cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10000)

            try:
                print("test")
                listener()
            except rospy.ROSInterruptException:
                pass


    else:
        rospy.loginfo("Le bouton A n'a pas été enfoncé.")




if __name__ == '__main__': 

    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10000)
    rospy.init_node('button_subscriber')
    rate = rospy.Rate(10) 
    try:
            # S'abonner au topic /button_pressed
            rospy.Subscriber('/button_pressed', Bool, button_callback)
            rate.sleep()
            rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Interruption de l'exécution du nœud button_subscriber.")
