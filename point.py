import math

def calc_angle(point1, point2, point3):
    # calculate angle between 1->2, 1->3
    a = Point.distance(point1, point2)
    b = Point.distance(point1, point3)
    c = Point.distance(point2, point3)
    return math.degrees(math.acos((a**2 + b**2 - c**2)/(2*a*b)))




class Point:
    '''
    class representing a single keypoint coordinate
    '''
    def __init__(self, x : float, y : float):
        self.x = x
        self.y = y

    def is_higher(self, other):
        '''
        Method checks whether this point is higher than a given point
        :param other: the Point instance compared to this one
        :return:  True if this point is higher than the other one, False otherwise
        '''
        return self.y < other.y

    def is_to_the_right(self, other):
        '''
        Method checks whether this point is further ou to the right in the image than a given point
        :param other: the Point instance compared to this one
        :return:  True if this point is more to the right than the other one, False otherwise
        '''
        return self.x > other.x

    @staticmethod
    def distance(point1, point2):
       """
       static method calculates the distance between two given points
       """
       return math.sqrt( (point1.x - point2.x)**2 + (point1.y - point2.y)**2 )

    @staticmethod
    def incline(point1, point2):
        if point1.x == point2.x and point1.y == point2.y:
            return 0
        dy = point1.y - point2.y
        dx = point1.x - point2.x
        return dy/dx

