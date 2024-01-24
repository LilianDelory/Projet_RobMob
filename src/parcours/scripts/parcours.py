#!/usr/bin/env python3

from compileall import compile_path
from os import path
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
import time


global cmd_vel_pub

# Variable globale pour stocker les points du chemin
path_points = []

#Compteur Global
compteur = 0

#Seuile de précision pour l'erreur de position
seuil = 0.1

#Position globale
pose  = [0,0,0]

#Gains pour loi de commande
K1 = 2
K2 = 2

#Paramètres du robot
L1 = 0.05

def odom_callback(msg):
    global pose
    position = msg.pose.pose.position
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion (orientation_list)

    pose = [position.x, position.y, yaw]

def markers_callback(msg):
    global path_points
    path_points = []
    for marker in msg.markers:
        for point in marker.points:
            path_points.append([point.x, point.y])
    print(path_points)


def commande(pose, path_points):
    global compteur
    global K1
    global K2
    global L1
    print(pose)
    
    xr = pose[0]
    yr = pose[1]
    theta = pose[2]

    erreur  = np.sqrt((xr - path_points[compteur][0])**2 + (yr - path_points[compteur][1])**2 )

    if erreur < seuil : 
        compteur += 1

    print("compteur :", compteur)
    v1 = -K1*(path_points[compteur][0]-xr)
    v2 = -K2*(path_points[compteur][1]-yr)
    v = [[v1],[v2]]


    rot = [[np.cos(theta), -L1*np.sin(theta)],[np.sin(theta), L1*np.cos(theta)]]
    inv_rot = np.linalg.inv(rot)
    u = np.dot(inv_rot,v)
    u1 = -u[0]
    u2 = -u[1]

    # Créer et publier la commande Twist
    cmd_vel_msg = Twist()
    cmd_vel_msg.linear.x = u1
    cmd_vel_msg.angular.z = u2
    print (cmd_vel_msg)
    cmd_vel_pub.publish(cmd_vel_msg)
    print(cmd_vel_msg)


def listener():
    while not rospy.is_shutdown():
        rospy.init_node('odom_listener', anonymous=True)

        # Subscribe to /odom topic
        rospy.Subscriber('/odom', Odometry, odom_callback)
        print(pose)
        commande(pose, path_points)
        rate = rospy.Rate(1) 
    rospy.spin()

if __name__ == '__main__': 

    # Subscribe to the topic publishing MarkerArray
    
    rospy.Subscriber('/Publication', MarkerArray, markers_callback)
    time.sleep(2)
    
    cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10000)

    try:
        listener()
    except rospy.ROSInterruptException:
        pass
