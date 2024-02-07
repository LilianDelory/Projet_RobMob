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
import tf

global cmd_vel_pub

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


def odom_callback():
    global pose
    listener = tf.TransformListener()

    try:
        listener.waitForTransform('/map', '/base_link', rospy.Time(0), rospy.Duration(1))
        (trans, rot) = listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        pose = [trans[0], trans[1], euler_from_quaternion(rot)[2]]
    except tf.Exception as e:
        rospy.logwarn("Failed to get the transformation: %s", str(e))




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
    rospy.init_node('odom_listener', anonymous=True)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown() and compteur < len(path_points):

        odom_callback()
        commande(pose, path_points)
        rate.sleep()

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
