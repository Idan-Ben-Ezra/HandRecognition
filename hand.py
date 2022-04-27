import math
from finger import *


class Hand:

    FINGERS_IN_HAND = 5

    def __init__(self, keypoints):
        self.angle = self.get_angle(keypoints)
        self.palm = Point(*keypoints[0]) # the first keypoint is of the hand's palm
        self.fingers = self.__fingers_from_keypoints(keypoints[1:])  # skipping the palm keypoint
        self.thumb, self.index_finger, self.middle_finger, self.ring_finger, self.pinky_finger = self.fingers




    def __fingers_from_keypoints(self, finger_keypoints):
        """
        Auxiliary method converts a list of the x,y coordinates of keypoints on fingers in a hand into a list of Finger objects
        :param finger_keypoints: a list of tuples - (x,y) representing the coordinates of all keypoints of the fingers
        :return: a list of Finger objects representing these keypoints
        """
        finger_xy = [finger_keypoints[i*Finger.POINTS_IN_FINGER:(i+1)*Finger.POINTS_IN_FINGER] for i in range(Hand.FINGERS_IN_HAND)]  # the x,y coordinates
            # ordered by 4 coordinates for each Finger
        return [Finger([Point(*coords) for coords in finger_coords], self.palm, self.angle) for finger_coords in finger_xy]  # converting the coordinates
            # to Point objects, and creating the Finger objects


    def get_hand_pose(self):
        """
        Method calculates the pose of the hand based on it's Finger's states. The returned 'state' code is in a `binary`
            format, meaning that the Fingers' states are just added one after the other to produce the code.
            e.g. if the Fingers states are: (Finger1:1,Finger2:3, Finger1:1, Finger1:2, Finger1:2), then the state is:
            13122
        :return: an int representing the hand pose
        """
        return int("".join(map(lambda finger1: str(finger1.bent_finger), self.fingers)))

    def get_angle(self, keypoints):
        keypoints = [Point(*i) for i in keypoints]
        return calc_angle(keypoints[0], keypoints[1], keypoints[5])




