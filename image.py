import math

from classes import *


class IoULoss(torch.nn.Module):
    """
    Intersection over Union Loss.
    IoU = Area of Overlap / Area of Union
    IoU loss is modified to use for heatmaps.
    """

    def __init__(self):
        super(IoULoss, self).__init__()
        self.EPSILON = 1e-6

    def _op_sum(self, x):
        return x.sum(-1).sum(-1)

    def forward(self, y_pred, y_true):
        inter = self._op_sum(y_true * y_pred)
        union = (
            self._op_sum(y_true ** 2)
            + self._op_sum(y_pred ** 2)
            - self._op_sum(y_true * y_pred)
        )
        iou = (inter + self.EPSILON) / (union + self.EPSILON)
        iou = torch.mean(iou)
        return 1 - iou


def get_heatmap_loss():
    mse = torch.nn.MSELoss()
    pdist = torch.nn.PairwiseDistance(p=2, eps=1e-10)

    def loss(heatmaps, labels):

        center = parse_heatmaps_tensor(heatmaps)
        # tensor flow mash grid - https://pytorch.org/docs/master/generated/torch.meshgrid.html
        # centre loss
        # we got the labels and the center of mass, we want to calc the dist between them.

        curr_loss = pdist(center, labels).mean(axis=-1) / MAX_DIST # sum the total loss among heatmaps in the same image
        # recheck
        # avg_loss = torch.mean(total_loss)  # get the avg loss among all images in batch
        # return avg_loss
        print("*****", curr_loss)
        return curr_loss


    def func(x, y):
        total_loss = 0
        if not isinstance(x, list):
            x = [x]
        for output in x:
            a = loss(output, y)
            total_loss += a
        return total_loss

    return loss

def combined_loss():
    HM_Loss = get_heatmap_loss()
    IoU_Loss = IoULoss()

    def loss(heatmap, labels):
        labels_hm, labels_center = labels
        # return (HM_Loss(heatmap, labels_center) + IoU_Loss.forward(heatmap, labels_hm))/2
        return IoU_Loss.forward(heatmap, labels_hm)
    return loss


device = get_default_device()
def main(model_file):

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
    labels = [float(i) for i in extract_labels("D:\magshimim\project\dataset\Data_Labels.csv")[int(IMG.split(".")[0])]]

    labels_hm = coordinate_to_heatmap_transform(labels)
    labels_hm = torch.FloatTensor(labels_hm)

    xkp = [label * HEATMAP_SIZE for label in labels][1::2]
    ykp = [label * HEATMAP_SIZE for label in labels][2::2]
    labels_center = list(zip(xkp, ykp))
    labels_center = torch.FloatTensor(labels_center)

    labels_center = torch.unsqueeze(labels_center, 0)
    image = cv2.imread(IMG)
    image = resize_transform(image)
    image = scale_transform(image)
    image = [image]
    image = torch.FloatTensor(image)
    data = image.permute(0, 3, 1, 2) # the current dimensions are of format-(batch, height, width, channels)
    IOU_Loss = IoULoss()
    HM_loss = get_heatmap_loss()
    combined = combined_loss()

    data, labels_hm, labels_center = to_device(data, device), to_device(labels_hm, device), to_device(labels_center, device)
    labels = (labels_hm, labels_center)
    for e in range(CURR_EPOCH, EPOCHS):
        train_loss = 0.0
        model.train()  # Optional when not using Model Specific layer
        for batch_num in tqdm(range(0, 35971//BATCH_SIZE)):
            optimizer.zero_grad()
            target = model.forward(data)[-1]
            # loss = IOU_Loss.forward(target, labels)  # calculating the loss for the predictions
            loss = combined(target, labels)
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