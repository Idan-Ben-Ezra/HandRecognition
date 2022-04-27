from main import *
import sys
from hand import *

def main():
    IMG = DATASET_PATH + "/" + input("enter image number:") + ".jpg"
    image = cv2.imread(IMG)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)

    key_points = [float(i) for i in
                  extract_labels("D:\magshimim\project\dataset\\Full_data.csv")[int(IMG.split(".")[0].split("/")[-1])]]
    xkp = key_points[1::2]
    ykp = key_points[2::2]
    labels = list(zip(xkp, ykp))
    assert key_points[0]
    hand = Hand(labels)
    pose = [int(i) for i in str(hand.get_hand_pose())]
    for num, val in enumerate(pose):

        printer_enum = {1: "bent", 2: "straight", 3: "pointing"}
        p = printer_enum[val]
        print(f'finger #{num+1} is ' + p)
    xkp = key_points[1::2]
    ykp = key_points[2::2]
    key_points = list(zip(xkp, ykp))
    for kp in key_points:
        image = cv2.circle(image, [int(kp[0] * 500), int(kp[1] * 500)], 3, (0, 255, 0), cv2.FILLED)
    cv2.imshow("fuck this", image)
    cv2.waitKey(0)
    cv2.destroyWindow("fuck this")

if __name__ == "__main__":
    main()