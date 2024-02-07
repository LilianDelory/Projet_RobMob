#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

def joy_callback(msg):
    # Logique pour convertir les données du joystick en commandes de mouvement
    # Exemple: Utiliser les axes du joystick pour contrôler la vitesse linéaire et angulaire

    
    linear_speed = msg.axes[1]  # Utilisez l'axe 1 pour la vitesse linéaire
    angular_speed = msg.axes[0]  # Utilisez l'axe 0 pour la vitesse angulaire

    # Créer un message Twist pour envoyer les commandes de mouvement
    twist_cmd = Twist()
    twist_cmd.linear.x = linear_speed*1.5
    twist_cmd.angular.z = angular_speed*1.5

    # Publier les commandes sur le topic /cmd_vel
    cmd_vel_pub.publish(twist_cmd)

    # Vérifier si le bouton A a été enfoncé
    if msg.buttons[0] == 1:  # bouton A est le premier dans la liste des boutons
        button_msg = True  # Le bouton A a été enfoncé
    else:
        button_msg = False  # Le bouton A n'a pas été enfoncé

    # Publier le message booléen sur un autre topic
    button_pub.publish(button_msg)



if __name__ == '__main__':
    try:
        print("Initialisation du nœud joy_teleop...")
        rospy.init_node('joy_teleop')

        # Définir le nom du topic du joystick (ajustez-le selon votre configuration)
        joy_topic = '/joy'

        # S'abonner au topic du joystick
        rospy.Subscriber(joy_topic, Joy, joy_callback)

        # Créer un éditeur pour publier les commandes de mouvement
        cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Créer un éditeur pour publier le message booléen
        button_pub = rospy.Publisher('/button_pressed', Bool, queue_size=10)


        # Boucle principale de ROS
        rate = rospy.Rate(10)  # 10 Hz
        rospy.loginfo("Attente de messages du joystick sur le topic %s", joy_topic)
        rospy.loginfo("Publier les commandes de mouvement sur le topic %s", '/cmd_vel')
        
        while not rospy.is_shutdown():
            # Ajoutez ici toute logique supplémentaire si nécessaire
            rate.sleep()

    except rospy.ROSInterruptException:
        rospy.loginfo("Interruption de l'exécution du nœud joy_teleop.")
