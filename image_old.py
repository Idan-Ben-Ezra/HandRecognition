from classes import *




def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_heatmap_loss():
    loss = torch.nn.MSELoss()
    def func(x, y):
        total_loss = 0
        if not isinstance(x, list):
            x = [x]
        for output in x:
            assert (output.shape == y.shape)

            total_loss += loss(output, y)

        return total_loss
    return func





def main(model_file):
    device = get_default_device()
    # function optimized to run on gpu
    loader = Loader(model_file)
    model = loader.load()
    model = model.cuda()

    # the optimizer and loss we use are identical to those used in the AWR paper
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    joint_keypoint_loss = get_joint_keypoint_loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    heatmap_loss = get_heatmap_loss()
    running_loss = 0.0
    labels = [float(i) for i in extract_labels("D:\magshimim\project\dataset\Data_Labels.csv")[int(IMG.split(".")[0])]][1:]
    labels = coordinate_to_heatmap_transform(labels)
    labels = torch.FloatTensor(labels)
    labels = labels.to(torch.float32)
    labels = torch.unsqueeze(labels, 0)
    image = cv2.imread(IMG)
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    data = image.permute(0, 3, 1, 2) # the current dimensions are of format-(batch, height, width, channels)
    print(labels.shape, data.shape)
    data, labels = to_device(data, device), to_device(labels, device)
    for e in range(CURR_EPOCH, EPOCHS):
        train_loss = 0.0
        model.train()  # Optional when not using Model Specific layer
        for batch_num in range(0, 35971, BATCH_SIZE):
            optimizer.zero_grad()
            target = model.forward(data)[-1]

            loss = heatmap_loss(target, labels)  # calculating the loss for the predictions
            loss.backward()  # performing backpropagation on the model
            optimizer.step()
            running_loss += loss.item()
            if batch_num % 100 == 99:
                print(" loss:", running_loss/100)
                train_loss += running_loss
                running_loss = 0
                loader.save(model)



# script.py model.bin new
if __name__ == "__main__":
    torch.cuda.empty_cache()
    assert len(sys.argv) <= 3
    model_file = DATASET_ROOT_DIR_PATH + "\\"

    if len(sys.argv) == 1:
        main(model_file + DEFAULT_FILENAME)
        exit(0)
    model_file += str(sys.argv[1])

    if str(sys.argv[-1]) == "new":
        loader = Loader(model_file)
        model = Model()
        loader.save(model)
    main(model_file)