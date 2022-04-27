from classes import *


# Training

def _init_dataset(data_path = DATASET_PATH):
    train_dataset = Dataset(data_path, LABEL_PATH,
                            transform=torchvision.transforms.Compose([resize_transform, scale_transform]),
                            label_transform=torchvision.transforms.Compose([coordinate_to_heatmap_transform]))
    return train_dataset

def _calc_loss(hg_outputs, y):
    pass
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def _init_dataloader():

    train_dataset = _init_dataset(DATASET_ROOT_DIR_PATH + "/train/Dataset")
    validation_dataset = _init_dataset(DATASET_ROOT_DIR_PATH + "/validation/Dataset")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, val_loader


def main(config):
    device = get_default_device()
    # function optimized to run on gpu
    loader = Loader(config["model"])
    model = loader.load()
    model = model.cuda()
    # until we fix this problem, we are going to use a single hourglass instead
    train_loader, val_loader = _init_dataloader()

    with open(config["lr"], "r") as file:
        learning_rate = float(file.readlines()[0])
    curr_epoch, total_epochs = config["curr_epoch"], config["max_epoch"]

    # the optimizer and loss we use are identical to those used in the AWR paper
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    joint_keypoint_loss = get_joint_keypoint_loss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6, verbose=True, threshold=0.00001)
    heatmap_loss = get_heatmap_loss()
    running_loss = 0.0
    with open(config["val_loss"], "r") as file:
        min_valid_loss = float(file.readlines()[0])
    loader_val = Loader(config["model"].split(".")[0] + "_val." + config["model"].split(".")[-1])
    print("min val loss is:", min_valid_loss)

    if config["loss"] == "get_combined_loss":
        get_loss = get_combined_loss()
    elif config["loss"] == "iou_loss":
        get_loss = IoULoss()
    else:
        raise("unknown loss function")
    for e in range(curr_epoch, total_epochs):

        train_loss = 0.0
        model.train()  # Optional when not using Model Specific layer
        for batch_num, batch in tqdm(enumerate(train_loader)):

            data, labels = batch
            labels = labels.to(torch.float32)
            data = data.permute(0, 3, 1, 2)  # the current dimensions are of format-(batch, height, width, channels)

            data, labels = to_device(data, device), to_device(labels, device)
            optimizer.zero_grad()
            target = model.forward(data)
            loss = get_loss.forward(target, labels)  # calculating the loss for the predictions
            loss.backward()  # performing backpropagation on the model
            optimizer.step()
            running_loss += loss.item()
            if batch_num % config["batch_print"] == config["batch_print"] - 1:
                loader.save(model)
                print(" loss:", running_loss / config["batch_print"], "lr:", get_lr(optimizer))
                train_loss += running_loss
                running_loss = 0
                if batch_num % config["scheduler_batch"] == config["scheduler_batch"] - 1:
                    lr_scheduler.step(loss.item())

        with open(os.path.join(DATASET_ROOT_DIR_PATH, config["lr"]), "w") as file:
            file.write(str(get_lr(optimizer)))
        valid_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer
        for data, labels in tqdm(val_loader):
            with torch.no_grad():
                data, labels = to_device(data, device), to_device(labels, device)
                labels = labels.to(torch.float32)
                data = data.permute(0, 3, 1, 2)  # the current dimensions are of format-(batch, height, width, channels)
                target = model.forward(data)
                loss = get_loss.forward(target, labels)  # calculating the loss for the predictions
                valid_loss += loss.item()

        print(
            f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
        with open(config["losses"] + "\\" + str(e) + ".txt", "w") as file:
            file.write(f'Epoch {e} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')
        print(
            f"validation loss from curr epoch: {valid_loss}, comparing to prev epoch: {min_valid_loss}"
        )
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            loader_val.save(model)
            with open(config["val_loss"], "w") as file:
                file.write(str(min_valid_loss) + "\nEpoch num: " + str(e))
        epoch_loader = Loader(config["model"].split(".")[0] + "_epochs" + str(e) + "." + config["model"].split(".")[-1])
        epoch_loader.save(model)


# script.py model.bin new
if __name__ == "__main__":
    torch.cuda.empty_cache()
    assert len(sys.argv) == 2
    with open(DATASET_ROOT_DIR_PATH + "\\" + sys.argv[1]) as f:
        config = json.loads(f.read())
    if config["new"].lower() == "true":
        loader = Loader(config["model"])
        model = Model()
        loader.save(model)
    main(config)