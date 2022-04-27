from consts import *
# LOSS

class get_heatmap_loss(torch.nn.Module):

    def __init__(self):
        super(get_heatmap_loss, self).__init__()

    def forward(self, x, y):
        total_loss = 0
        if not isinstance(x, list):
            x = [x]
        for output in x:
            assert(output.shape == y.shape)
            z = (output - y).float()
            mse_mask = (torch.abs(z) < 0.01).float()
            l1_mask = (torch.abs(z) >= 0.01).float()
            mse = mse_mask * z
            l1 = l1_mask * z
            total_loss += torch.mean(self._calculate_MSE(mse)*mse_mask)
            total_loss += torch.mean(self._calculate_L1(l1)*l1_mask)

        return total_loss

    def _calculate_MSE(self, z):
        return 0.5 *(torch.pow(z, 2))

    def _calculate_L1(self, z):
        return 0.01 * (torch.abs(z) - 0.005)

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
        def loss(y_pred, y_true):
            inter = self._op_sum(y_true * y_pred)
            union = (
                    self._op_sum(y_true ** 2)
                    + self._op_sum(y_pred ** 2)
                    - self._op_sum(y_true * y_pred)
            )
            iou = (inter + self.EPSILON) / (union + self.EPSILON)
            iou = torch.mean(iou)
            return 1 - iou
        total_loss = 0
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        for output in y_pred:
            total_loss += loss(output, y_true)
        return total_loss


class get_combined_loss(torch.nn.Module):

    def __init__(self):
        super(get_combined_loss, self).__init__()

    def first(self, y_pred, y_true):
        total_loss = 0
        loss = IoULoss()
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        for output in y_pred:
            total_loss += loss.forward(output, y_true)

        return total_loss

    def second(self, y_pred, y_true):
        loss = get_heatmap_loss()
        return loss(y_pred, y_true)

    def forward(self, y_pred, y_true):
        loss = (self.first(y_pred, y_true) + self.second(y_pred, y_true)) / 2

        return loss
def __joint_keypoint_loss(loss_func):
    """
    Decorator returns the function which calculates the loss
    :param loss_func: the loss function applied to the output
    :return: the inner function, which calculates the loss
    """

    def calculate_loss(outputs, labels):
        if outputs[HAND_DETECTION_INDEX] != labels[HAND_DETECTION_INDEX]:
            return INVALID_DETECTION_LOSS_VALUE
        return loss_func(outputs, labels)

    return calculate_loss


def get_joint_keypoint_loss():
    return __joint_keypoint_loss(torch.nn.SmoothL1Loss())

# HMs to KP


def get_heatmaps(key_points):
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

        heatmaps = blur_heatmaps(heatmaps)
    return heatmaps

coordinate_to_heatmap_transform = get_heatmaps

def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], BLUR_GRID, BLUR_DESC)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred
'''def get_batch_heatmaps(labels):
    """
    Function returns the heatmaps for a given batch of labels
    """
    print(labels.shape, labels, len(labels.shape))
    if len(labels.shape) == 1:  # if the batch consits only of a single record
      print("su")
      return get_heatmaps(labels)
    return torch.FloatTensor([get_heatmaps(single_image_label) for single_image_label in labels])
'''


def parse_heatmaps(heatmaps):
    """
    Heatmaps is a numpy array
    Its size - (batch_size, n_keypoints, img_size, img_size)
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(axis=-1).sum(axis=-1)
    sums = np.expand_dims(sums, [2, 3])
    normalized = heatmaps / sums
    x_prob = normalized.sum(axis=2)
    y_prob = normalized.sum(axis=3)

    arr = np.tile(np.float32(np.arange(0, 64)), [batch_size, 21, 1])
    x = (arr * x_prob).sum(axis=2)
    y = (arr * y_prob).sum(axis=2)
    keypoints = np.stack([x, y], axis=-1)
    return keypoints / 64

def parse_heatmaps_tensor(heatmaps):
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(axis=-1).sum(axis=-1)
    sums = torch.unsqueeze(sums, 2).unsqueeze(3)
    normalized = to_device(heatmaps / sums, device=get_default_device())
    x_prob = normalized.sum(axis=2)
    y_prob = normalized.sum(axis=3)
    arr = to_device(torch.tile(torch.arange(0, HEATMAP_SIZE, dtype=torch.float32), [batch_size, N_KEYPOINTS, 1]), device=get_default_device())
    x = (arr * x_prob).sum(axis=2)
    y = (arr * y_prob).sum(axis=2)
    keypoints = torch.stack([x, y], axis=-1)
    return keypoints / HEATMAP_SIZE

#
# CLASSES
#

# Bottleneck


class Bottleneck(torch.nn.Module):
    """
    The Bottleneck block used in the Hourglass architecture
    The block contains 3 conv layers:
    a 1 by 1 layer, a 3 by 3 layer, and a 1 by 1 layer
    (batch normalization are added for increased efficiency)
    the layer is generally used to imporve efficiency:
    it's much faster-has less learnable parameters- than a simple 3 by 3 convolution
    """

    PADDING = 'same'  # each layer is padded so that the output is of same size as the input

    def __init__(self, in_channels, out_channels, to_expand_channels=False, to_downsample=False):
        '''
        Class ctor
        :param in_channels: The amount of channels that the input has
        :param out_channels: The amount of channels that the output has.
        :param to_expand_channels: Whether the output's channels should be the 'out_channels' value.
          if not, output's channels eqauls 'in_channels'
        :param to_downsample: Whether the output's height should be reduced by a scale of 2,
          if not, the output height is the same as the input height
        '''
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                     padding=Bottleneck.PADDING)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                     padding=Bottleneck.PADDING)
        self.conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1,
                                     padding=Bottleneck.PADDING)
        # iconv3 output channels should be the same as the un_channles, because of the sum operation with the residual layer(has in_channels)
        self.relu = torch.nn.ReLU()
        self.batch_normalization1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.batch_normalization2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.batch_normalization3 = torch.nn.BatchNorm2d(num_features=in_channels)

        self.to_expand_channels = to_expand_channels
        if self.to_expand_channels:
            self.__init_channel_expansion(in_channels, out_channels)

        self.to_downsample = to_downsample
        if self.to_downsample:
            self.__init_downsample(in_channels,
                                   in_channels)  # applied on the residual, the amount of channels shouldn't be changed

        self.in_c = in_channels
        self.out_c = out_channels

    def __init_channel_expansion(self, in_channels, out_channels):
        '''
        Auxiliary method initalizes the channels expansion block, which is responsible for increasing the amount of channels-
        is responsible for making the Bottleneck's output contain 'out_channels' amount of channels
        :return: None
        '''
        self.expand_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                           padding=Bottleneck.PADDING)
        self.expand_batch_normalization = torch.nn.BatchNorm2d(num_features=out_channels)

    def __init_downsample(self, in_channels, out_channels):
        '''
        Auxiliary method initalizes the dwonsampling layer, used for decreasing the output height of the Bottleneck
        is used by the Hourglass for residual layer
        :return: None
        '''
        self.downsample = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                                          stride=STRIDE)

    @staticmethod
    def get_out_height(in_height):
        '''
        Auxiliary method outputs the height of the bottleneck's output
        :param in_height: the input's height
        :return: the new output's height
        '''
        # height1 = calc_out_size(in_height, )
        # note that in this implementation the bottleneck keeps the output the same size
        return in_height

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # print("Channels: ", self.in_c, self.out_c)
        residual = x
        # print("X - ", x.shape)
        y = self.relu(self.conv1(x))
        y = self.batch_normalization1(y)
        # print("Y1 - ", y.shape)
        y = self.relu(self.conv2(y))
        y = self.batch_normalization2(y)
        # print("Y2 - ",y.shape)
        y = self.conv3(y)
        y = self.batch_normalization3(y)
        # print("Y3 - ",y.shape)
        # the bottleneck block performs a skip connection from the input to the last layer

        y += residual

        if self.to_downsample:  # decreasing the height of the output, so that it fits the next layer
            y = self.downsample(y)

        if self.to_expand_channels:  # expanding the amount of channels outputted by the bottleneck,
            # so that the output is of depth 'out_channels'
            y = self.expand_conv(y)
            y = self.expand_batch_normalization(y)

        y = self.relu(y)

        return y


# Connector


class Connector(torch.nn.Module):
    """
    This layer connects stacked Hourglass networks together -
    In the model architecture, the backbone network consists of multiple hourglass models stacked on each other.
    This layer converts the output of one hourglass into the input of the next horuglass
    """

    def __init__(self, in_channels=HOURGLASS_OUTPUT_CHANNELS, out_channels=RGB_CHANNELS,
                 heatmap_channels=HEATMAP_CHANNELS):
        '''
        class ctor, note that it's input is the hourglass output,
        while it's output is the next hourglass' input
        :param in_channels: The amount of channels in the hourglass' output
        :param out_channels: THe amount of channels a hourglass receives as input
        '''
        super(Connector, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=heatmap_channels, kernel_size=1)

        self.conv2 = torch.nn.Conv2d(in_channels=heatmap_channels, out_channels=out_channels, kernel_size=1)

        self.conv3 = torch.nn.Conv2d(in_channels=heatmap_channels, out_channels=out_channels, kernel_size=1)
        # the conv applied after loss is calculated on heatmaps

        self.deconv_layer = torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
                                                     kernel_size=DECONV_KERNEL_SIZE,
                                                     stride=DECONV_STRIDE)  # the layer upscales the input so that the next horuglass
        # receives a 128*128 sized input(instead of a 64*64 sized heatmap)

    def __call__(self, x, network_input):
        return self.forward(x, network_input)

    def forward(self, x, network_input):
        '''
        The forward method of this layer
        :param x: the output of the previous hourglass
        :param network_input: the input of the previous hourglass
        :return: a tuple containing: the input to the next hourglass,
            the heatmaps on which a loss is going to be calculated(losses are calculated on each hourglass' output separately)
        '''
        # print("X - ", x.shape, "net inp - ", network_input.shape)
        x = self.conv1(x)
        heatmap_preds = x  # the heatmap predictions on which a loss is going to be calculated
        # print("hm - ", heatmap_preds.shape)
        y = self.conv2(x)
        residual_y = self.conv3(heatmap_preds)
        # print("Y - ", y.shape, "res y - ", residual_y.shape)

        y = y + residual_y
        upscaled_y = self.deconv_layer(
            y)  # the output is upscaled so that it aligns with the next hourglass' dimensionality
        upscaled_y += network_input
        # print("Finished connecting")

        return upscaled_y, heatmap_preds


# Conv Block



DOWNSCALE_PADDING = 3

class ConvBlock(torch.nn.Module):
    """
  The basic Conv block that is used by the horuglass model
  The block itself contains a 7*7 conv layer, a batch normalization layer and a Relu activation function
  """

    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING):
        super(ConvBlock, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=SAME_PADDING)
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=3)
        self.bn_layer = torch.nn.BatchNorm2d(num_features=out_channels, momentum=BATCH_NORM_MOMENTUM)
        self.relu = torch.nn.ReLU()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = self.conv_layer(x)
        #print(x.shape)
        #cv2_imshow(x.detach().numpy()[0][0])

        x = self.bn_layer(x)
        x = self.relu(x)

        return x

# Dataset

def extract_labels(filename):
    """

  :param filename:
  :return:
  """
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        listed =list(reader)
    return listed


class Dataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, labels_path, transform=(lambda x: x), label_transform=(lambda x: x)):
        """
    Class constructor

    """
        super(Dataset, self).__init__()
        self.labels = extract_labels(labels_path)
        self.images = sorted(os.listdir(
            images_dir), key=get_id_from_name)  # please make sure images are sorted with key correctly
        self.images_dir = images_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.images[idx]
        image = cv2.imread(os.path.join(self.images_dir, image_name))
        image = self.transform(image)
        label = self.labels[get_id_from_name(image_name)]
        label = [float(i) for i in label]
        label = torch.tensor(label)
        label = self.label_transform(label)

        return image, label


# Loader

class Loader:
    def __init__(self, filename):
        self._filename = filename

    def save(self, model):
        torch.save(model.state_dict(), self._filename)

    def load(self):
        model = Model()
        model.load_state_dict(torch.load(self._filename))
        return model

# Model


class DecoderBlock(torch.nn.Module):
    """
  The basic block in the decoder part of the hourglass.
  Contains 1)an upscaling of the previous layer's input,
  and 2)a bottleneck that receives as input a residual value.
  The two are added together.
  """

    def __init__(self, in_dimensions, out_dimensions, kernel_size=DECONV_KERNEL_SIZE):
        '''
    class constructor
    :param in_dimensions: the dimensions of the given input, in the format: [N, Channels, Height, Width]
    :param out_dimension: the dimensions of the expected input, in the format: [N, Channels, Height, Width]
    '''
        super(DecoderBlock, self).__init__()
        assert len(in_dimensions) == 4 and len(
            out_dimensions) == 4  # checking that the batch size is part of the input dimensions

        self.deconv_layer = torch.nn.ConvTranspose2d(in_channels=in_dimensions[1], out_channels=out_dimensions[1],
                                                     kernel_size=kernel_size,
                                                     stride=DECONV_STRIDE)  # the deconv layer allows us to increase the feature's hieght amd width(while also learning new weights)
        self.bn_layer = torch.nn.BatchNorm2d(num_features=out_dimensions[1], momentum=BATCH_NORM_MOMENTUM)
        self.relu = torch.nn.ReLU()
        self.bottleneck = Bottleneck(in_channels=out_dimensions[1], out_channels=out_dimensions[1],
                                     to_expand_channels=True)
        self.in_c = in_dimensions[1]
        self.out_c = out_dimensions[1]

    def __call__(self, x, residual):
        return self.forward(x, residual)

    def forward(self, x, residual):
        '''
    :param x: the output of the previous layer
    :param residual: the residual output fed to this layer
    '''
        # print("IN channels: ", self.in_c, "OUT channels: ", self.out_c)
        # print("Decode: X - ", x.shape, "res - ", residual.shape)
        x = self.deconv_layer(x)
        # print("Decode: X - ", x.shape, "res - ", residual.shape)
        x = self.relu(self.bn_layer(x))
        # print("Decode: X - ", x.shape, "res - ", residual.shape)
        residual = self.bottleneck(residual)
        # print("Decode: X - ", x.shape, "res - ", residual.shape)
        y = x + residual

        return y


class EncoderBlock(torch.nn.Module):
    """
  The basic block in the decoder part of the hourglass.
  Contains 1)an upscaling of the previous layer's input,
  and 2)a bottleneck that receives as input a residual value.
  The two are added together.
  """

    def __init__(self, in_channels, out_channels):
        '''
    class constructor
    :param in_channels: The amount of channels that the input has
    :param out_channels: The amount of channels that the output has.
    '''
        super(EncoderBlock, self).__init__()
        assert in_channels * 2 == out_channels  # the depth of the features is increased times 2 by each block
        self.bottleneck = Bottleneck(in_channels=in_channels, out_channels=out_channels, to_expand_channels=True)
        self.max_pool = torch.nn.MaxPool2d(MAX_POOL_FILTER, stride=STRIDE, padding=0, dilation=1, return_indices=False,
                                           ceil_mode=False)

    @staticmethod
    def get_out_height(in_height):
        '''
    Auxiliary method outputs the height of the layer's final output
    :param in_height: the input's height
    :return: the new output's height
    '''
        height1 = int(in_height / 2)  # the max pool divides the height by 2
        height2 = Bottleneck.get_out_height(height1)

        return height2

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        '''
    :return: a tuple containing: the input to the encoding layer,
            the input to the residual layer connected to the encoder
    '''
        y_residual = self.bottleneck(x)  # this output is used by the residual layers
        y = self.max_pool(y_residual)  # this output is use by the next encoding layer

        return y, y_residual


class FeatureExtractor(torch.nn.Module):
    """
    This layer extracts the deepest features in the Hourglass network, its features are the smallets and in height, and
    biggest in deoth.   After this layer the features are decoded into a heatmap.
    """

    def __init__(self, in_channels, out_channels):
        '''
        class ctor
        :param in_channels: The amount of channels that the input has. If input is of shape [N, C, H, W]- the second dim= channels
        :param out_channels: The amount of channels that the output has.
        '''
        super(FeatureExtractor, self).__init__()
        self.bottleneck1 = Bottleneck(in_channels=in_channels, out_channels=in_channels * 2, to_expand_channels=True)
        # need to check whether the in_channles, out_channels value is correct
        self.bottleneck2 = Bottleneck(in_channels=in_channels * 2, out_channels=in_channels * 2)
        # need to check whether the in_channles, out_channels value is correct
        self.bottleneck3 = Bottleneck(in_channels=in_channels * 2, out_channels=in_channels * 2,
                                      to_expand_channels=True)
        # need to check whether the in_channles, out_channels value is correct

        self.upsample = torch.nn.Upsample(scale_factor=UPSCALE)
        # A upsample is used to increase the extracted features' height so that it matches the reidual feature's height
        # there is a difference in height because a max pool layer is applied by the decoder only after the residual connection

        self.residual_max_pool = torch.nn.MaxPool2d(MAX_POOL_FILTER, stride=STRIDE, padding=0, dilation=1,
                                                    return_indices=False, ceil_mode=False)
        # A max pool is used to reduce the reidual feature's height so that it matches the features' height
        # there is a difference in height because a max pool layer is applied by the decoder only after the residual connection

    def __call__(self, x, residual_y):
        return self.forward(x, residual_y)

    def forward(self, x, residual_y):
        '''
        The forward method of the final feature extraction layer
        :param x: the output of the previous layer
        :param residual_y: the residual output (after it has been passed through a bottleneck)
        :return: the final extracted feature(the most high-level features)
        '''
        # print("extractor: ", x.shape, residual_y.shape)
        y = self.bottleneck1(x)
        # print("extractor: ", y.shape, residual_y.shape)
        y = self.bottleneck2(y)
        y = self.bottleneck3(y)
        # print("extractor: ", y.shape, residual_y.shape)

        residual_y = self.residual_max_pool(
            residual_y)  # decreasing the residual's height so that it matches y's height
        # y = self.upsample(y)
        # print("shape ", residual_y.shape, y.shape)
        y = y + residual_y

        return y




class FirstLayer(torch.nn.Module):
    """
    The first layer on the architecture
    """

    def __init__(self, out_channels, in_channels=RGB_CHANNELS):
        '''
        class ctor
        :param in_channels: The amount of channels that the input has
        :param out_channels: The amount of channels that the output has.
        '''
        super(FirstLayer, self).__init__()
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=out_channels)
        self.bottleneck = Bottleneck(in_channels=out_channels, out_channels=out_channels)

        self.max_pool = torch.nn.MaxPool2d(MAX_POOL_FILTER, stride=STRIDE, padding=0, dilation=1, return_indices=False,
                                           ceil_mode=False)
        self.upsample = torch.nn.Upsample(scale_factor=UPSCALE)
    @staticmethod
    def get_out_height(in_height):
        '''
        Auxiliary method outputs the height of the layer's final output
        :param in_height: the input's height
        :return: the new output's height
        '''
        height1 = calc_out_size(in_height, 0, STRIDE, KERNEL_SIZE)
        height2 = Bottleneck.get_out_height(height1)

        return height2

    def forward(self, x):
        x = self.conv_block(x)

        y_residual = self.bottleneck(x)

        y = y_residual
        y_residual = self.upsample(y_residual)
        #y = self.max_pool(y_residual)

        return y, y_residual


class Hourglass(torch.nn.Module):
    """
    The Backbone Hourglas-'Encoder-Decoder' architecture, based on the paper - https://arxiv.org/pdf/1603.06937.pdf.
    The network is based upon extensive use of Residual Bottleneck layers, which are used for both encoding and decoding features.
    First the image is encoded into features, which are then decoded(using residual connections to the encoded layers) into a heatmap.
    A more in depth explanation- https://towardsdatascience.com/using-hourglass-networks-to-understand-human-poses-1e40e349fa15
    """

    def __init__(self, in_channels=RGB_CHANNELS, out_channels=HEATMAP_CHANNELS, extract_heatmaps=False):
        '''
        Class ctor for hourglass network
        :param in_channels: the amount of channels(depth) of the input to the network, default value is 3-RGB image
        :param out_channels: the amount of channels(depth) of the network - only if it outputs heatmaps directly,
        meaning that it's the last layer - , is equal to the amount of heatmaps outputted, which is the amount of keypoints
        :param extract_heatmaps: bool value representing whether the network outputs heatmaps directly, basically
        representing whether the netowrk is the last hourglass in the backbone
        '''
        super(Hourglass, self).__init__()
        # the encoding layers of the model
        self.encoder_layer1 = FirstLayer(in_channels=in_channels,
                                         out_channels=64)  # the first layer foesn;t have a max pooling layer,
        # instead the dimensions are reduced by the Conv operation(which isn't padded) *unsure whether it;s the same as the paper
        # self.height1 = FirstLayer.get_out_height(INPUT_HEIGHT)  # h = 64
        self.encoder_layer2 = EncoderBlock(in_channels=64, out_channels=128)
        # self.height2 = EncoderBlock.get_out_height(self.height1) # 32
        self.encoder_layer3 = EncoderBlock(in_channels=128, out_channels=256)
        # self.height3 = EncoderBlock.get_out_height(self.height2) # 16
        self.encoder_layer4 = EncoderBlock(in_channels=256, out_channels=512)
        # self.height4 = EncoderBlock.get_out_height(self.height3) # 8
        self.feature_extractor = FeatureExtractor(in_channels=512,
                                                  out_channels=1024)  # out_channels may need to be changed to a different value

        # the residual layers of the model
        # it's unclear whether a residual layer consists a single or multiple bottlenecks

        # the residual layer's Height is always bigger than the corresponding decoder layer height, as during encoding,
        # the residual is outputted before the input goes through Max pooling, meaning that the residual's height isn't decreased
        # (while the feature's height is).
        # This is solved by downsampling the output in the residual layer itself, so that it height matches the decoder
        self.residual1 = Bottleneck(in_channels=64, out_channels=128, to_expand_channels=True, to_downsample=True)
        self.residual2 = Bottleneck(in_channels=128, out_channels=256, to_expand_channels=True, to_downsample=True)
        self.residual3 = Bottleneck(in_channels=256, out_channels=512, to_expand_channels=True, to_downsample=True)

        self.residual4 = Bottleneck(in_channels=512, out_channels=1024,
                                    to_expand_channels=True)  # this residual's output is sent to the feature extractor

        # after every applied Conv filter- the image shrinks,
        # if the images dimensions were (C, H, W), and padding=p, stride=s, kernel_size=f
        # then the output would be - Hout = lower((H+2*p-f)/s + 1)
        # the decoding layers of the network
        '''
        self.decoder_layer1 = DecoderBlock((BATCH_SIZE, 512, self.height3, self.height3), (BATCH_SIZE, 256, self.height2, self.height2)) # height-16*2= 32
        self.decoder_layer2 = DecoderBlock((BATCH_SIZE, 256, self.height3, self.height3), (BATCH_SIZE, 128, self.height2, self.height2)) # height-32*2= 64
        self.decoder_layer3 = DecoderBlock((BATCH_SIZE, 128, self.height4, self.height4), (BATCH_SIZE, 64, self.height3, self.height3))'''
        self.decoder_layer1 = DecoderBlock((BATCH_SIZE, 1024, 1, 1), (BATCH_SIZE, 512, 1, 1))  # height-8*2 =16
        self.decoder_layer2 = DecoderBlock((BATCH_SIZE, 512, 1, 1), (BATCH_SIZE, 256, 1, 1))  # height-16*2= 32
        self.decoder_layer3 = DecoderBlock((BATCH_SIZE, 256, 1, 1), (BATCH_SIZE, 128, 1, 1))  # height-32*2= 64

        self.heatmap_extractor = None
        if extract_heatmaps:
            self.heatmap_extractor = torch.nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1)
        # there might need to be an additional layer which converts the output into 42 heatmaps, or that layer
        # is inside the connector_layer(on which's output the loss is calculated for each hourglass)

        # whether the channels=HEATMAP_CHANNELS assignment is correct needs to be checked
        # print(self.height1, self.height2, self.height3, self.height4)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):

        # the input is passed through each decoder layer,
        # and the output of each decoder is also passed to the residual
        layer1, res1 = self.encoder_layer1(x)
        res1 = self.residual1(res1)
        # the encoding blocks have 2 outputs: one for the next encoding layer and one for the residual layer
        layer2, res2 = self.encoder_layer2(layer1)
        res2 = self.residual2(res2)
        layer3, res3 = self.encoder_layer3(layer2)
        res3 = self.residual3(res3)
        layer4, res4 = self.encoder_layer4(layer3)
        res4 = self.residual4(res4)
        # the deepest res and deepest features are passed to the final feature extractor
        features = self.feature_extractor(layer4, res4)

        # print("Heights: ", res1.shape[-1], res2.shape[-1], res3.shape[-1], res4.shape[-1])
        # print("Heights: ", layer1.shape[-1], layer2.shape[-1], layer3.shape[-1], layer4.shape[-1])

        # print("Channels: ", layer1.shape[1],layer2.shape[1], layer3.shape[1], layer4.shape[1], features.shape[1] )
        # now the features and the corresponding residual are passed through the decoding layers
        # print("Deco1 input - ", features.shape, res3.shape)
        decoded1 = self.decoder_layer1(features, res3)
        decoded2 = self.decoder_layer2(decoded1, res2)
        final_decoded = self.decoder_layer3(decoded2, res1)

        # heatmaps = self.heatmap_extractor(decoded3)
        # it's unclear whether each hourglass outputs heatmaps(on which the loss is calculated),
        #  or whether the heatmaps are only created by the connector layer between two stacked hourglasses
        if self.heatmap_extractor:
            final_decoded = self.heatmap_extractor(final_decoded)

        return final_decoded

class Model(torch.nn.Module):
    """
    THe AWR model class

    """

    def __init__(self, num_hourglass=HOURGLASS_LAYERS):
        super(Model, self).__init__()
        self.hourglass_layers = torch.nn.ModuleList(
            [Hourglass(in_channels=RGB_CHANNELS, out_channels=HEATMAP_CHANNELS) for i in range(num_hourglass - 1)])
        # initializing all layers bu the last one, which extacts it's heatmaps directly (without a connector layer)
        self.hourglass_layers.append(
            Hourglass(in_channels=RGB_CHANNELS, out_channels=HEATMAP_CHANNELS, extract_heatmaps=True))
        self.connector_layers = torch.nn.ModuleList([Connector(in_channels=HOURGLASS_OUTPUT_CHANNELS,
                                                               out_channels=RGB_CHANNELS,
                                                               heatmap_channels=HEATMAP_CHANNELS)
                                                     for i in range(
                num_hourglass - 1)])  # there are n-1 connectors for n hourglass layers

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        hg_output = []
        # passing the input through all hourglass layers
        for i, connector in enumerate(self.connector_layers):
            horuglass_input = x
            x = self.hourglass_layers[i](x)  # input is passed through the Hourglass net
            x, heatmaps = connector(x, horuglass_input)  # then the input is passed to the connector_layer,
            # which outputs the input for the next layer, and the heatmap predictions
            hg_output.append(heatmaps)
        x = self.hourglass_layers[-1](x)  # the output is passed to the last layer
        hg_output.append(x)
        return hg_output  # returning all the outputted heatmaps, as loss is calculated on each of them('immediate supervision')

