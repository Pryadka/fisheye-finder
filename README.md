# fisheye-finder
Using a fisheye camera to find objects.

The goal of the project is to analyze the image obtained from a fisheye camera 360 by 220 degrees in order to determine the direction of the desired object, in this situation a drone. The idea is to mechanically rotate the camera in the found direction until the object is in the center of the camera. The result of the neural network should be a one-hot vector with the probabilities that the object is in this sector.

![regions](https://github.com/user-attachments/assets/4b305bfb-63cd-4419-a6d6-9a964febbb77)

Example:

![pred_1841](https://github.com/user-attachments/assets/ceede7bb-a837-4941-91bd-0f2bfc2f5057)

