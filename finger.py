from point import *
import enum


class FingerStatesEnum(enum.Enum):
    '''
    Enum class representing the possible states of a single finger
    '''
    BENT = 1
    STRAIGHT = 2
    POINTING = 3  # when the finger is half bent or is pointing at the camera
    #BENT, STRAIGHT, POINTING = range(1,4)

class Finger:
    '''
    Class representing a single finger in the hand image
    '''
    POINTS_IN_FINGER = 4

    def __init__(self, points, pivot, angle):
        self.lowest, self.mid1, self.mid2, self.highest = points  # initializing the coordinates of the lowest, middle, highest keypoints
        self.pivot = pivot  # keypoint number 0, the one on the palm of the hand, not the finger
        self.__calc_distances()  # calculating the distances between each keypoint and the pivot
        self.bent_finger = self.__is_bent_finger()  # saving the finger state
        self.angle = angle  # keypoint number 1, the thumb's lowest


    def __calc_distances(self):
        self.highest_to_pivot_dist = Point.distance(self.highest, self.pivot)
        self.mid2_to_pivot_dist = Point.distance(self.mid2, self.pivot)
        self.mid1_to_pivot_dist = Point.distance(self.mid1, self.pivot)
        self.lowest_to_pivot_dist = Point.distance(self.lowest, self.pivot)
        self.distances = (self.highest_to_pivot_dist, self.mid2_to_pivot_dist, self.mid1_to_pivot_dist, self.lowest_to_pivot_dist)
        self.min_dist = min(*self.distances)

    def __is_pointing(self):
        '''
        Method checks whether the current finger is half bent or is pointing at the camera
        :return:
        '''
        # we know if a finger is bent based on performing regression on all points (except pivot).
        # another option - look at mid2 and highest line compared to mid2. if line is below the point at a certain amount - finger is bent
        # we are not using depth images, so for us there's no knowing whether a finger is "facing" the camera or not.
        # since we cant tell if some parts of the finger are closer to the camera we have no sense of direction,
        # making telling whether a finger is pointing or just laid horizontally impossible by looking only at the finger.
        #
        # in order to try and cope with the lack of information we cant use other parts of the hand,
        # especially the 0-point, aka the wrist point, the ground for getting some sense of depth.
        # to be as accurate as possible, we can say the hand / finger is laid down if and only if
        # the wrist point is at the same level and is "close enough".

        # in order to say whether two points are close or not, once again, without using depth, we have to get creative.
        # as we cannot use a 3d distance calculation between the x, y, depth values of the points we can
        # try and ignore the depth, which will cause less accurate results,
        # but cover-up for when finger seems to be impossible.

        # a thing we need to discuss when talking about distance is what is close and what is far.
        # the simplest way would be to define a const value of percentage from the image,
        # but this way is mildly incorrect as the distance of the hand, and hence it's size, may defer from image to image.

        # another option is to calculate the distance from the pivot to the base of the first finger,
        # the thumb, as an indication for the intire hand.

        # one more accurate way for determine closeness if to look at the 0-point and the base of one of the fingers.
        # in this way, we can say that the bigger the distance between the pivot and the lowest point,
        # measured in <self.lowest_to_pivot_dist>. although when a hand is laid down,
        # the short distance may indicate that the hand is way further than the camera than it actually is.

        # it seems that because of different angles of the hand there can't be
        # a *single* distance that can clarify the angel,
        # we can try a combination of some of the distances to get "part" of the hand angle.
        # we won't be able to get the total angle because we are using 2d images, and the hand's angle is 3d.
        # to decide the hand we take the incline of the pivot and the thumb-lowest, and the one from the pivot and the
        # current finger lowest.
        # this way, the narrower the angle, the more flattened the hand is, and hence,
        # the shorter the distances from point to point have to be in order to assume they are close.

        # in this situation, we can't determine what is the angle for the thumb,
        # and since the fingers are all at the same angle
        # we can calculate the angle between pivot->thumb-lowest and pivot-> index-lowest.

        # implementation:
        # first we need to figure out the two inclines:


        # a safer way would be to caclulate the angles of the fractions inside the hand, between the fingers.





        return False

    def __is_bent_finger(self):
        """
        Method checks the state of this finger - is it bent, straight or pointing
        :return: the enu value corresponding to the finger's state
        """
        if self.__is_pointing():
            return FingerStatesEnum.POINTING.value
        """
        if self.min_dist == self.highest_to_pivot_dist or self.min_dist == self.mid2_to_pivot_dist:
            # then the finger is bent
            return FingerStatesEnum.BENT.value

        if self.min_dist == self.lowest_to_pivot_dist:
            # then the finger definitely isn't bent
            return FingerStatesEnum.STRAIGHT.value
        """

        angle1 = calc_angle(self.lowest, self.pivot, self.mid1)
        angle2 = calc_angle(self.mid1, self.lowest, self.mid2)
        angle3 = calc_angle(self.mid2, self.mid1, self.highest)
        if angle2 < 120 or angle3 < 120:
            return FingerStatesEnum.BENT.value
        if angle1 < 120:
            return FingerStatesEnum.POINTING.value
        return FingerStatesEnum.STRAIGHT.value



