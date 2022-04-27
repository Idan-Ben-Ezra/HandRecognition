from model_testing import *
import test_algorythm
def main(config):
    print("presentation of our project:")

    print("showing pose determination algorythm:")
    while True:
        test_algorythm.main()
        if input("should repeat? (y/n): ").lower() != "y":
            break

    print("showing current_model:")
    while True:
        test2()
        if input("should repeat? (y/n): ").lower() != "y":
            break
    print("showing heatmaps progress:")
    IMG = DATASET_PATH + "/" + input("enter image number:") + ".jpg"
    labels = [float(i) for i in
              extract_labels("D:\magshimim\project\dataset\Full_Data.csv")[int(IMG.split(".")[0].split("/")[-1])]]
    our_heatmaps = np.asarray(get_heatmaps(labels))
    for epoch in tqdm(range(1, config["curr_epoch"])):
        loader = Loader(DATASET_ROOT_DIR_PATH + "\\model_epochs" + str(epoch) + ".bin")
        model = loader.load()
        plt.figure(1)
        image = cv2.imread(IMG)
        image = resize_transform(image)
        image = scale_transform(image)
        image = [image]
        image = torch.FloatTensor(image)
        image = image.permute(0, 3, 1, 2)

        heatmaps_model = model(image)[-1][0]
        plt.imshow(heatmaps_model[10].detach().numpy())
        plt.figure(2)
        plt.imshow(our_heatmaps[10])
        plt.show()
        cv2.waitKey(0)




if __name__ == "__main__":
    with open(DATASET_ROOT_DIR_PATH + "\\" + sys.argv[1]) as f:
        config = json.loads(f.read())
    main(config)
