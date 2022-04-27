# we want to find images with labels that are in the first and last quarters of the image, meaning not in the middle
import os.path

from main import *

def get_images(images_dir = DATASET_PATH, labels_path = LABEL_PATH):
    labels = extract_labels(labels_path)
    images = sorted(os.listdir(
            images_dir), key=get_id_from_name)
    assert len(images) == len(labels)
    return {"images": images, "labels": labels}


FILTER_DEPTH = 20

SECOND_DRAW_COLOR = (0, 0, 255)
RADIUS = 3


def test_edge():
    new_dataset = {"images": [], "labels": [], "numbers": []}
    dataset = get_images()
    for index in range(len(dataset["images"])):
        labels = [float(i) for i in dataset["labels"][index]]
        if labels[0] == 0:
            continue
        cnt = 0
        total = 0
        for num, label in enumerate(labels[1:]):
            if not (0.25 < label < 0.75):
                total = (num, label)
                cnt += 1
        if cnt > FILTER_DEPTH:
            new_dataset["images"].append(dataset["images"][index])
            new_dataset["labels"].append(dataset["labels"][index])
            new_dataset["numbers"].append(total)

    for (image, labels) in zip(new_dataset["images"], new_dataset["labels"]):
        def extract_labels(filename, idx):
            """

          :param filename:
          :return:
          """
            with open(filename, newline='') as f:
                reader = csv.reader(f)
                listed = list(reader)
            return listed[idx]

        img = image
        image = os.path.join(DATASET_PATH, image)
        image = cv2.imread(image)
        image = resize_transform(image)
        image = scale_transform(image)
        image = [image]
        image = torch.FloatTensor(image)
        image = image.permute(0, 3, 1, 2)
        loader = Loader(FILE)
        model = loader.load()
        heatmaps = model(image)[-1]

        labels = parse_heatmaps_tensor(heatmaps)
        image = cv2.imread(img)
        image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
        labels = labels[0]
        points = [[int(i[0] * 500), int(i[1] * 500)] for i in labels]
        key_points = [float(i) for i in
                      extract_labels("D:\magshimim\project\dataset\Data_Labels.csv",
                                     int(IMG.split(".")[0].split("/")[-1]))]
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








def filter(image):
    return ()



if __name__ == "__main__":
    main()