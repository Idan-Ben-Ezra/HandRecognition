import torch
import numpy as np
import torch
torch.manual_seed(0)

import cv2
import csv
import matplotlib.pyplot as plt
plt.grid(True)
IMAGE_WIDTH = 128
HEATMAP_SIZE = 64
MAX_VALUE = 1
BLUR_GRID = (5, 5)
IMAGE_HEIGHT = IMAGE_WIDTH
DRAW_COLOR = (255, 0, 255)
SECOND_DRAW_COLOR = (0, 0, 255)
RADIUS = 3
import sys
from main import *
from my_camera import my_camera


FILE = "D:\magshimim\project\dataset\\model.bin"




def get_images(images_dir = DATASET_PATH, labels_path = LABEL_PATH):
    def extract_labels(filename):
        """

      :param filename:
      :return:
      """
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            listed = list(reader)
        return listed
    labels = extract_labels(labels_path)
    images = sorted(os.listdir(
            images_dir), key=get_id_from_name)
    assert len(images) == len(labels)
    return {"images": images, "labels": labels}


FILTER_DEPTH = 20

SECOND_DRAW_COLOR = (0, 0, 255)
RADIUS = 3


def test_edge():
    new_dataset = {"images": [], "labels": []}
    dataset = get_images()
    for index in range(len(dataset["images"])):
        labels = [float(i) for i in dataset["labels"][index]]
        if labels[0] == 0:
            continue
        cnt = 0
        total = 0
        for num, label in enumerate(labels[1:]):
            if not (0.25 < label < 0.75):
                cnt += 1
        if cnt > FILTER_DEPTH:
            new_dataset["images"].append(dataset["images"][index])
            new_dataset["labels"].append(dataset["labels"][index])

    for (image, labels) in zip(new_dataset["images"], new_dataset["labels"]):


        image = os.path.join(DATASET_PATH, image)
        img = image
        image = cv2.imread(image)
        image = resize_transform(image)
        image = scale_transform(image)
        image = [image]
        image = torch.FloatTensor(image)
        image = image.permute(0, 3, 1, 2)
        loader = Loader(FILE)
        model = loader.load()
        heatmaps = model(image)[-1]

        # labels = parse_heatmaps_tensor(heatmaps)
        labels = find_max(heatmaps)
        image = cv2.imread(img)
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
        labels = labels[0]
        points = [[int(i[0] * 500), int(i[1] * 500)] for i in labels]
        key_points = [float(i) for i in
                      extract_labels("D:\magshimim\project\dataset\Full_Data.csv")[int(img.split(".")[0].split("\\")[-1])]]
        xkp = key_points[1::2]
        ykp = key_points[2::2]
        key_points = list(zip(xkp, ykp))
        quarter = 500 // 4

        image = cv2.line(image, (quarter, quarter), (3 * quarter, quarter), SECOND_DRAW_COLOR, 5)
        image = cv2.line(image, (3 * quarter, quarter), (3 * quarter, 3 * quarter), SECOND_DRAW_COLOR, 5)
        image = cv2.line(image, (3 * quarter, 3 * quarter), (quarter, 3 * quarter), SECOND_DRAW_COLOR, 5)
        image = cv2.line(image, (quarter, quarter), (quarter, 3 * quarter), SECOND_DRAW_COLOR, 5)
        for num, point in enumerate(points):
            kp = key_points[num]
            image = cv2.circle(image, [int(kp[0] * 500), int(kp[1] * 500)], RADIUS, SECOND_DRAW_COLOR, cv2.FILLED)
            image = cv2.circle(image, point, RADIUS, DRAW_COLOR, cv2.FILLED)

        cv2.imshow("fuck this", image)
        cv2.waitKey(0)

def find_max(heatmaps):
    a = heatmaps.detach().numpy()
    result = []
    for batch in a:
        lst = []
        for hm in batch:
            lst.append(np.asarray(np.unravel_index(hm.argmax(), hm.shape)[::-1])/64)
        result.append(lst)
    return result

def test2():
    IMG = DATASET_PATH + "/" + input("enter image number:") + ".jpg"
    image = cv2.imread(IMG)
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    image = image.permute(0, 3, 1, 2)
    loader = Loader(FILE)
    model = loader.load()
    heatmaps = model(image)[-1]

    # labels = parse_heatmaps_tensor(heatmaps)
    labels = find_max(heatmaps)
    image = cv2.imread(IMG)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    labels = labels[0]
    points = [[int(i[0] * 500), int(i[1] * 500)] for i in labels]
    key_points = [float(i) for i in
               extract_labels("D:\magshimim\project\dataset\\Full_data.csv")[int(IMG.split(".")[0].split("/")[-1])]]
    xkp = key_points[1::2]
    ykp = key_points[2::2]
    key_points = list(zip(xkp, ykp))
    quarter = 500 // 4

    for num, point in enumerate(points):
        kp = key_points[num]
        image = cv2.circle(image, [int(kp[0] * 500), int(kp[1] * 500)], RADIUS, (0, 255, 0), cv2.FILLED)
        image = cv2.circle(image, point, RADIUS, DRAW_COLOR, cv2.FILLED)

        cv2.imshow("fuck this", image)
        cv2.waitKey(0)
    cv2.destroyWindow("fuck this")


def test3():
    IMG = DATASET_PATH + "/" + input("enter image number:") + ".jpg"
    image = cv2.imread(IMG)
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    image = image.permute(0, 3, 1, 2)
    loader = Loader(FILE)
    model = loader.load()
    heatmaps_model = model(image)[-1]
    heatmaps_model = heatmaps_model
    print(parse_heatmaps_tensor(heatmaps_model))
    labels = [float(i) for i in
                  extract_labels("D:\magshimim\project\dataset\Full_Data.csv")[int(IMG.split(".")[0].split("/")[-1])]]
    our_heatmaps = np.asarray(get_heatmaps(labels))

    for i in range(len(heatmaps_model[0])):

        plt.imshow(heatmaps_model[0][i].detach().numpy())
        cv2.waitKey(0)
        plt.show()

        plt.imshow(our_heatmaps[i])
        plt.show()
        cv2.waitKey(0)

    print(parse_heatmaps(heatmaps_model.detach().numpy()))
    print(labels)

def test4():
    IMG = "D:\\MYIMAGES\\Camera Roll\\a.jpg"
    image = cv2.imread(IMG)
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    image = image.permute(0, 3, 1, 2)
    loader = Loader(FILE)
    model = loader.load()
    heatmaps = model(image)[-1]

    labels = parse_heatmaps_tensor(heatmaps)
    # labels = find_max(heatmaps)
    image = cv2.imread(IMG)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    labels = labels[0]
    points = [[int(i[0] * 500), int(i[1] * 500)] for i in labels]
    """
    for num, point in enumerate(points):
        image = cv2.circle(image, point, RADIUS, DRAW_COLOR, cv2.FILLED)

        cv2.imshow("fuck this", image)
        cv2.waitKey(0)
    """
    for i in heatmaps[0]:
        plt.imshow(i.detach().numpy())
        plt.show()
        cv2.waitKey(0)

def test_live():
    camera = my_camera()
    # model = Hourglass(extract_heatmaps=True)
    # model.load_state_dict(torch.load("hourglass_model_built_from_existing.pth"))
    loader = Loader(FILE)
    model = loader.load()
    while True:
        frame = camera.get_photo()
        image = resize_transform(frame)
        image = scale_transform(image)
        image = [image]
        image = torch.FloatTensor(image)
        image = image.permute(0, 3, 1, 2)
        heatmaps_model = model(image)[-1]
        labels = parse_heatmaps_tensor(heatmaps_model)
        labels = labels[0]
        points = [[int(i[0] * 500), int(i[1] * 500)] for i in labels]
        frame = cv2.resize(frame, (500, 500))
        quarter = 500 // 4


        for point in points:
            frame = cv2.circle(frame, point, RADIUS, DRAW_COLOR, cv2.FILLED)

        camera.display_window(frame, "1")
        if camera.check_key_presssed("1"):
            camera.destroy_window("1")
            break



def test_loss():
    def get_heatmaps_v2(key_points):
        """
        Creates 2D heatmaps from keypoint locations for a single image
        Input: array of size N_KEYPOINTS x 2
        Output: array of size N_KEYPOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
        """
        xkp = key_points[1::2]
        ykp = key_points[2::2]
        kp = list(zip(xkp, ykp))
        heatmaps = np.zeros([N_KEYPOINTS, HEATMAP_SIZE, HEATMAP_SIZE])
        if key_points[0]:  # the first value in the list states whether theres a hand or not.
            #                we want the default heatmap to be empty rather than (0, 0) labels.
            for k, (x, y) in enumerate(kp):
                x, y = int(x * HEATMAP_SIZE), int(y * HEATMAP_SIZE)
                if (0 <= x < HEATMAP_SIZE) and (0 <= y < HEATMAP_SIZE):
                    heatmaps[k, int(y), int(x)] = 1

            heatmaps = blur_heatmaps_v2(heatmaps)
        return heatmaps

    def blur_heatmaps_v2(heatmaps):
        """Blurs heatmaps using GaussinaBlur of defined size"""
        heatmaps_blurred = heatmaps.copy()
        for k in range(len(heatmaps)):
            if heatmaps_blurred[k].max() == 1:
                heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (9, 9), BLUR_DESC+10)
                heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
        return heatmaps_blurred
    from copy import deepcopy as dc
    labels = [float(i) for i in extract_labels("D:\magshimim\project\dataset\Data_Labels.csv", int(IMG.split(".")[0]))]
    real_heatmaps = torch.FloatTensor(get_heatmaps(labels))
    labels = [i + 10/64 for i in labels]
    fake_heatmaps = torch.FloatTensor(get_heatmaps(labels)) - 1
    loss_calc = get_combined_loss()
    print("loss:", loss_calc.forward(real_heatmaps, fake_heatmaps))
    for i in range(len(real_heatmaps)):

        plt.imshow(real_heatmaps[i])
        cv2.waitKey(0)
        plt.show()

        plt.imshow(fake_heatmaps[i])
        plt.show()
        cv2.waitKey(0)

def test_mass():
    pdist = torch.nn.PairwiseDistance(p=2, eps=1e-10)

    labels = [float(i) for i in extract_labels("D:\magshimim\project\dataset\Data_Labels.csv", int(IMG.split(".")[0]))]
    labels = to_device(torch.FloatTensor(list(zip(labels[1::2], labels[2::2]))), get_default_device())
    img = cv2.imread(IMG)
    image = img
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    image = image.permute(0, 3, 1, 2)
    loader = Loader(FILE)
    model = loader.load()
    heatmaps = model(image)[-1]
    centers = parse_heatmaps_tensor(heatmaps)
    total_length = 0
    centers = centers[0]
    print(len(labels))
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    for num, point in enumerate(centers):
        kp = labels[num]
        image = img
        image = cv2.circle(image, [int(kp[0] * 500), int(kp[1] * 500)], RADIUS, DRAW_COLOR, cv2.FILLED)
        image = cv2.circle(image, [int(point[0] * 500), int(point[1] * 500)], RADIUS, SECOND_DRAW_COLOR, cv2.FILLED)
        curr_length = pdist(point, kp)
        print("length:", curr_length)
        total_length += curr_length
        plt.imshow(image)
        cv2.waitKey(0)
        plt.show()

    print("----summary----")
    print("received total length:", (total_length/N_KEYPOINTS)/MAX_DIST)
    print("estimated total length:", pdist(centers, labels).mean(axis=-1) / MAX_DIST)


def test_conv():
    a = torch.zeros(8, 21, 64, 64)
    b = np.zeros([8, 21, 64, 64])
    a = a.unsqueeze(2).unsqueeze(3)
    b = np.expand_dims(b, (2, 3))
    print("torch:", a.shape)
    print("numpy:", b.shape)
    print(False in (a.detach().numpy() == b))

def test_print():
    import math
    x=0.6
    print(math.degrees(math.acos(x)))
    print(math.degrees(math.acos(-abs(x))))

if __name__ == "__main__":
    if str(sys.argv[-1]) == "2":
        test2()
    elif str(sys.argv[-1]) == "4":
        test4()
    elif str(sys.argv[-1]) == "loss":
        test_loss()
    elif str(sys.argv[-1]) == "mass":
        test_mass()
    elif str(sys.argv[-1]) == "conv":
        test_conv()
    elif str(sys.argv[-1]) == "live":
        test_live()
    elif str(sys.argv[-1]) == "print":
        test_print()
    elif str(sys.argv[-1]) == "edge":
        test_edge()
    else:
        test3()
