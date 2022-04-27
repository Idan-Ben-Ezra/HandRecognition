"""
* file: my_camera.py
* writer: idan ben ezra
* date: 23/9/2021

* the file is a class allowing usage of cv2 as a camera object
* that can capture and display frames from the webcam.
*
* frame capturing: capture photo from the webcam (get_photo)
* window controlling: display a window (display_window), destroy a window (destroy_window, destroy_all, destroy_window_on_key)
* helpers: checking if key is pressed (check_key_pressed)
* frame editing: resizing (resize_frame), converting to numpy array(get_numpy_array)
"""
from consts import *
DEFAULT_SIZE = IMAGE_WIDTH
class my_camera:

    def __init__(self):
        '''class constructor'''
        self._camera = cv2.VideoCapture(0)

    def get_photo(self):
        """
        takes photo from the camera and returns the frame if okay
        :return: None
        """
        return_value, frame = self._camera.read()
        if not return_value:
            print("failed to grab frame")
            exit(1)
        return frame

    def display_window(self, frame, window_name):
        """
        displays the frame given in a window named window_name
        :param frame: (numpy.ndarray) the frame displayed
        :param window_name: (str) the name of the window
        :return: None
        """
        cv2.imshow(window_name, frame)

    def destroy_all(self):
        """
        destroys all the opencv active windows
        :return: None
        """
        cv2.destroyAllWindows()

    def destroy_window(self, window_name):
        """
        destroys a window by name
        :param window_name: (str) the name of the destroyed window
        :return: None
        """
        try:
            cv2.destroyWindow(window_name)
        except cv2.error as e:
            print(f"no window named '{window_name}' was found active")

    def destroy_window_on_key(self, window_name, key):
        """
        destroys a window by name after key is pressed
        Note: will stop the run of the program if not started as a thread
        :param window_name: (str) the name of the destroyed window
        :param key: (str) the key to destroy the window
        :return: None
        """
        if not isinstance(key, str) or len(key) != 1: #checking that the key is string and 1 character long
            print("only 1 key allowed!")
            return
        while True:
            if self.check_key_presssed(key):
                self.destroy_window(window_name)
                break

    def check_key_presssed(self, key):
        """
        checks if the key given was pressed
        :param key: (str) the key checked to be pressed
        :return: (bool) was the key pressed
        """
        if not isinstance(key, str) or len(key) != 1: # checking that the key is string and 1 character long
            print("only 1 key allowed!")
            return False
        return cv2.waitKey(1) & 0xFF == ord(key)

    def resize_frame(self, frame, width = DEFAULT_SIZE, height = DEFAULT_SIZE):
        """
        resizes a given frame to width and height resolution
        :param frame: (numpy.ndarray) the frame resized
        :param width: (int) the new frame's width
        :param height: (int) the new frame's height
        :return: (numpy.ndarray) the resized frame
        """
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return resized

    def get_numpy_array(self, frame, width = DEFAULT_SIZE, height = DEFAULT_SIZE):
        """
        resizes the frame and converts it to numpy 3d array
        :param frame: (numpy.ndarray) the frame resized
        :param width: (int) the new frame's width
        :param height: (int) the new frame's height
        :return: (numpy.ndarray) the resized frame
        """
        return np.asarray(self.resize_frame(frame, width, height))



    def __del__(self):
        """
        destructor for the class. destroys all the active windows and releases the webcam
        :return: None
        """
        self.destroy_all()
        del(self._camera)

