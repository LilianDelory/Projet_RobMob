#!/usr/bin/env python3

import rospy
import cv2
import os
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
import numpy as np

def occupancy_grid_callback(data):
    # Extract data from OccupancyGrid message
    width = data.info.width
    height = data.info.height
    data_array = np.array(data.data, dtype=np.int8)
    resolution = data.info.resolution
    origin = data.info.origin
    print("resolution: ", resolution)
    print("origin: ", origin)

    # Reshape data to 2D array
    occupancy_grid = np.reshape(data_array, (height, width))

    # Convert occupancy grid to an 8-bit image (0: free, 100: occupied)
    image = (occupancy_grid * 2.55).astype(np.uint8)

    # Extract the center region (800x800) from the 4000x4000 image
    center_x = width // 2
    center_y = height // 2
    half_size = 400  # Half of the desired size (800/2)

    #The image is cropped only if the size if > 800x800
    if width > 800 and height > 800:
        cropped_image = image[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

    else:
        cropped_image = image


    # Display or save the cropped image as needed
    cv2.imshow("Cropped Occupancy Grid Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # To save the cropped image
    cv2.imwrite("my_occupancy_grid_image.png", cropped_image)

    # Save the map using map_saver
    os.system('rosrun map_server map_saver -f my_occupancy_grid')

def main():
    rospy.init_node('occupancy_grid_to_image_node', anonymous=True)

    # Subscribe to the occupancy grid topic
    rospy.Subscriber("/map", OccupancyGrid, occupancy_grid_callback)

    # Wait for the ROS node to finish
    rospy.spin()

if __name__ == '__main__':
    main()
