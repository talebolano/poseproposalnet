import numpy as np
import cv2
import math
import torch.nn.functional as F
import random
from scipy import ndimage
from torchvision.transforms import Resize


def resize_point(point, in_size, out_size):
    """Adapt point coordinates to the rescaled image space.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray or list of arrays:
        Points rescaled according to the given image shapes.

    """
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    if isinstance(point, np.ndarray):
        out_point = point.copy()
        out_point[:,  0] = y_scale * point[:, 0]
        out_point[:, 1] = x_scale * point[:, 1]
        # out_point = out_point.tolist()
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] = y_scale * pnt[:, 0]
            out_pnt[:, 1] = x_scale * pnt[:, 1]
            out_point.append(out_pnt)
    return out_point


def resize(image, keypoints, bbox, size):
    H, W,_= image.shape
    new_h, new_w = size
    image = cv2.resize(image, (new_w, new_h),interpolation=cv2.INTER_CUBIC)

    keypoints = [
        resize_point(points, (H, W), (new_h, new_w))
        for points in keypoints
    ]

    new_bbox = []
    for x, y, bw, bh in bbox:
        [[y, x]] = resize_point(np.array([[y, x]]), (H, W), (new_h, new_w))
        bw *= new_w / W
        bh *= new_h / H
        new_bbox.append(np.array([x, y, bw, bh],dtype='float32'))
    return image, keypoints, new_bbox


def scale_fit_short(image, keypoints, bbox, scale): #(int(min(h,w)*1.25))
    _, H, W = image.shape
    newH,newW = int(H*scale),int(W*scale)
    new_image = cv2.resize(image, (newW,newH),interpolation=cv2.INTER_CUBIC) # kuang gao
    new_keypoints = [scale * k for k in keypoints]
    new_bbox = [scale * np.asarray(b) for b in bbox]
    return new_image, new_keypoints, new_bbox


def augment_image(image):
    """color augmentation"""


    method = np.random.choice(
            ['random_distort', 'nonechance'],
            p=[0.5, 0.5],
        )

    if method == 'random_distort':
        image = random_distort(image, contrast_low=0.3, contrast_high=2)
        return image
    else:
        return image


def random_distort(
        img,
        brightness_delta=32,
        contrast_low=0.5, contrast_high=1.5,
        saturation_low=0.5, saturation_high=1.5,
        hue_delta=18):

    #cv_img = img[::-1].transpose((1, 2, 0)).astype(np.uint8)

    def convert(img, alpha=1, beta=0):
        img = img.astype(float) * alpha + beta
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)

    def brightness(cv_img, delta):
        if random.randrange(2):
            return convert(
                cv_img,
                beta=random.uniform(-delta, delta))
        else:
            return cv_img

    def contrast(cv_img, low, high):
        if random.randrange(2):
            return convert(
                cv_img,
                alpha=random.uniform(low, high))
        else:
            return cv_img

    def saturation(cv_img, low, high):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 1] = convert(
                cv_img[:, :, 1],
                alpha=random.uniform(low, high))
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    def hue(cv_img, delta):
        if random.randrange(2):
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
            cv_img[:, :, 0] = (
                cv_img[:, :, 0].astype(int) +
                random.randint(-delta, delta)) % 180
            return cv2.cvtColor(cv_img, cv2.COLOR_HSV2BGR)
        else:
            return cv_img

    cv_img = brightness(img, brightness_delta)

    if random.randrange(2):
        cv_img = contrast(cv_img, contrast_low, contrast_high)
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
    else:
        cv_img = saturation(cv_img, saturation_low, saturation_high)
        cv_img = hue(cv_img, hue_delta)
        cv_img = contrast(cv_img, contrast_low, contrast_high)

    return cv_img  # uint8


def crop(img, y_slice, x_slice, copy=False):
    ret = img.copy() if copy else img
    return ret[y_slice, x_slice,:]


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    def relu(x): return max(0, x)
    w = relu(min(x0 + w0, x1 + w1) - max(x0, x1))
    h = relu(min(y0 + h0, y1 + h1) - max(y0, y1))
    return w * h


def translate_bbox(bbox, size, y_offset, x_offset):
    cropped_H, cropped_W = size
    new_bbox = []
    for x, y, w, h in bbox:
        x_shift = x + x_offset
        y_shift = y + y_offset
        is_intersect = intersection([0, 0, cropped_W, cropped_H], [x_shift, y_shift, w, h])
        if is_intersect:
            xmin = max(0, x_shift)
            ymin = max(0, y_shift)
            xmax = min(cropped_W, x_shift + w)
            ymax = min(cropped_H, y_shift + h)
            wnew = xmax - xmin
            hnew = ymax - ymin
            new_bbox.append([xmin, ymin, wnew, hnew])
        else:
            new_bbox.append([x_shift, y_shift, w, h])
    return new_bbox


def translate_point(point, y_offset=0, x_offset=0):
    """Translate points.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray:
        Points modified translation of an image.

    """

    if isinstance(point, np.ndarray):
        out_point = point.copy()

        out_point[:,0] += y_offset
        out_point[:,1] += x_offset
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] += y_offset
            out_pnt[:, 1] += x_offset
            out_point.append(out_pnt)
    return out_point


def crop_all_humans(image, keypoints, bbox, is_labeled):
    H, W, _ = image.shape
    aspect = W / H
    param = {}
    if len(keypoints) == 0:
        param['do_nothing'] = True
        return image, keypoints, bbox

    kymax = max([np.max(ks[l, 0]) for l, ks in zip(is_labeled, keypoints)])
    kxmax = max([np.max(ks[l, 1]) for l, ks in zip(is_labeled, keypoints)])
    kymin = min([np.min(ks[l, 0]) for l, ks in zip(is_labeled, keypoints)])
    kxmin = min([np.min(ks[l, 1]) for l, ks in zip(is_labeled, keypoints)])

    bxmax = max([b[0] + b[2] for b in bbox])
    bymax = max([b[1] + b[3] for b in bbox])
    bxmin = min([b[0] for b in bbox])
    bymin = min([b[1] for b in bbox])

    ymax = max(kymax, bymax)
    xmax = max(kxmax, bxmax)
    ymin = min(kymin, bymin)
    xmin = min(kxmin, bxmin)

    if (xmax + xmin) / 2 < W / 2:
        x_start = random.randint(0, max(0, int(xmin)))
        y_start = random.randint(0, max(0, int(ymin)))
        y_end = random.randint(min(H, int(ymax)), H)
        ylen = y_end - y_start
        xlen = aspect * ylen
        x_end = min(W, int(x_start + xlen))
        x_slice = slice(x_start, x_end, None)
        y_slice = slice(y_start, y_end, None)
    else:
        x_end = random.randint(min(int(xmax), W), W)
        y_end = random.randint(min(int(ymax), H), H)
        y_start = random.randint(0, max(0, int(ymin)))
        ylen = y_end - y_start
        xlen = aspect * ylen
        x_start = max(0, int(x_end - xlen))
        x_slice = slice(x_start, x_end, None)
        y_slice = slice(y_start, y_end, None)

    cropped = crop(image, y_slice=y_slice, x_slice=x_slice, copy=True)
    cropped_H, cropped_W, _ = cropped.shape
    param['x_slice'] = x_slice
    param['y_slice'] = y_slice
    if cropped_H <= 50 or cropped_W <= 50:
        """
        This case, for example, cropped_H=0 will cause an error when try to resize image
        or resize small image to insize will cause low resolution human image.
        To avoid situations, we will stop crop image.
        """
        param['do_nothing'] = True
        return image, keypoints, bbox
    image = cropped

    keypoints = [
        translate_point(
            points, x_offset=-x_slice.start, y_offset=-y_slice.start)
        for points in keypoints
    ]

    bbox = translate_bbox(
        bbox,
        size=(cropped_H, cropped_W),
        x_offset=-x_slice.start,
        y_offset=-y_slice.start,
    )

    return image, keypoints, bbox


def _sample_parameters(size, scale_ratio_range, aspect_ratio_range):
    H, W = size
    for _ in range(10):
        aspect_ratio = random.uniform(
            aspect_ratio_range[0], aspect_ratio_range[1])
        if random.uniform(0, 1) < 0.5:
            aspect_ratio = 1 / aspect_ratio
        # This is determined so that relationships "H - H_crop >= 0" and
        # "W - W_crop >= 0" are always satisfied.
        scale_ratio_max = min((scale_ratio_range[1],
                               H / (W * aspect_ratio),
                               (aspect_ratio * W) / H))

        scale_ratio = random.uniform(
            scale_ratio_range[0], scale_ratio_range[1])
        if scale_ratio_range[0] <= scale_ratio <= scale_ratio_max:
            return scale_ratio, aspect_ratio

    # This scale_ratio is outside the given range when
    # scale_ratio_max < scale_ratio_range[0].
    scale_ratio = random.uniform(
        min((scale_ratio_range[0], scale_ratio_max)), scale_ratio_max)
    return scale_ratio, aspect_ratio


def image_random_sized_crop(img,
                      scale_ratio_range=(0.08, 1),
                      aspect_ratio_range=(3 / 4, 4 / 3),
                      return_param=False, copy=False):
    """Crop an image to random size and aspect ratio.

    The size :math:`(H_{crop}, W_{crop})` and the left top coordinate
    :math:`(y_{start}, x_{start})` of the crop are calculated as follows:

    + :math:`H_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\times a}}\\rfloor`
    + :math:`W_{crop} = \\lfloor{\\sqrt{s \\times H \\times W \
        \\div a}}\\rfloor`
    + :math:`y_{start} \\sim Uniform\\{0, H - H_{crop}\\}`
    + :math:`x_{start} \\sim Uniform\\{0, W - W_{crop}\\}`
    + :math:`s \\sim Uniform(s_1, s_2)`
    + :math:`b \\sim Uniform(a_1, a_2)` and \
        :math:`a = b` or :math:`a = \\frac{1}{b}` in 50/50 probability.

    Here, :math:`s_1, s_2` are the two floats in
    :obj:`scale_ratio_range` and :math:`a_1, a_2` are the two floats
    in :obj:`aspect_ratio_range`.
    Also, :math:`H` and :math:`W` are the height and the width of the image.
    Note that :math:`s \\approx \\frac{H_{crop} \\times W_{crop}}{H \\times W}`
    and :math:`a \\approx \\frac{H_{crop}}{W_{crop}}`.
    The approximations come from flooring floats to integers.

    .. note::

        When it fails to sample a valid scale and aspect ratio for ten
        times, it picks values in a non-uniform way.
        If this happens, the selected scale ratio can be smaller
        than :obj:`scale_ratio_range[0]`.

    Args:
        img (~numpy.ndarray): An image array. This is in CHW format.
        scale_ratio_range (tuple of two floats): Determines
            the distribution from which a scale ratio is sampled.
            The default values are selected so that the area of the crop is
            8~100% of the original image. This is the default
            setting used to train ResNets in Torch style.
        aspect_ratio_range (tuple of two floats): Determines
            the distribution from which an aspect ratio is sampled.
            The default values are
            :math:`\\frac{3}{4}` and :math:`\\frac{4}{3}`, which
            are also the default setting to train ResNets in Torch style.
        return_param (bool): Returns parameters if :obj:`True`.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns only the cropped image.

        If :obj:`return_param = True`,
        returns a tuple of cropped image and :obj:`param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_slice** (*slice*): A slice used to crop the input image.\
            The relation below holds together with :obj:`x_slice`.
        * **x_slice** (*slice*): Similar to :obj:`y_slice`.

            .. code::

                out_img = img[:, y_slice, x_slice]

        * **scale_ratio** (float): :math:`s` in the description (see above).
        * **aspect_ratio** (float): :math:`a` in the description.

    """
    H, W ,_ = img.shape
    scale_ratio, aspect_ratio =\
        _sample_parameters(
            (H, W), scale_ratio_range, aspect_ratio_range)

    H_crop = int(math.floor(np.sqrt(scale_ratio * H * W * aspect_ratio)))
    W_crop = int(math.floor(np.sqrt(scale_ratio * H * W / aspect_ratio)))
    y_start = random.randint(0, H - H_crop)
    x_start = random.randint(0, W - W_crop)
    y_slice = slice(y_start, y_start + H_crop)
    x_slice = slice(x_start, x_start + W_crop)

    img = img[y_slice, x_slice,:]

    if copy:
        img = img.copy()
    if return_param:
        params = {'y_slice': y_slice, 'x_slice': x_slice,
                  'scale_ratio': scale_ratio, 'aspect_ratio': aspect_ratio}
        return img, params
    else:
        return img


def translate_point(point, y_offset=0, x_offset=0):
    """Translate points.

    This method is mainly used together with image transforms, such as padding
    and cropping, which translates the top left point of the image
    to the coordinate :math:`(y, x) = (y_{offset}, x_{offset})`.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        y_offset (int or float): The offset along y axis.
        x_offset (int or float): The offset along x axis.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray:
        Points modified translation of an image.

    """

    if isinstance(point, np.ndarray):
        out_point = point.copy()

        out_point[:, 0] += y_offset
        out_point[:, 1] += x_offset
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] += y_offset
            out_pnt[:, 1] += x_offset
            out_point.append(out_pnt)
    return out_point


def random_sized_crop(image, keypoints, bbox):
    image, param = image_random_sized_crop(
        image,
        scale_ratio_range=(0.5, 5),
        aspect_ratio_range=(0.75, 1.3333333333333333),
        return_param=True
    )

    keypoints = [
        translate_point(points,
                                   x_offset=-param['x_slice'].start,
                                   y_offset=-param['y_slice'].start
                                   )
        for points in keypoints
    ]

    cropped_H, cropped_W,_ = image.shape

    bbox = translate_bbox(
        bbox,
        size=(cropped_H, cropped_W),
        x_offset=-param['x_slice'].start,
        y_offset=-param['y_slice'].start,
    )

    return image, keypoints, bbox


def rotate_point(point_yx, angle, center_yx):
    offset_y, offset_x = center_yx
    shift = point_yx - center_yx
    shift_y, shift_x = shift[:, 0], shift[:, 1]
    cos_rad = np.cos(np.deg2rad(angle))
    sin_rad = np.sin(np.deg2rad(angle))
    qx = offset_x + cos_rad * shift_x + sin_rad * shift_y
    qy = offset_y - sin_rad * shift_x + cos_rad * shift_y
    return np.array([qy, qx]).transpose()


def rotate_image(image, angle):
    rot = ndimage.rotate(image, angle, axes=(1,0), reshape=False)
    # disable image collapse
    rot = np.clip(rot, 0, 255)
    return rot


def random_rotate(image, keypoints, bbox):
    angle = np.random.uniform(-15, 15)
    new_keypoints = []
    center_yx = np.array(image.shape[:2]) / 2
    for points in keypoints:
        rot_points = rotate_point(np.array(points),
                                  angle,
                                  center_yx)
        new_keypoints.append(rot_points)

    new_bbox = []
    for x, y, w, h in bbox:

        points = np.array(
            [
                [y, x],
                [y, x + w],
                [y + h, x],
                [y + h, x + w]
            ]
        )

        rot_points = rotate_point(
            points,
            angle,
            center_yx
        )
        xmax = np.max(rot_points[:, 1])
        ymax = np.max(rot_points[:, 0])
        xmin = np.min(rot_points[:, 1])
        ymin = np.min(rot_points[:, 0])
        # x,y,w,h
        new_bbox.append([xmin, ymin, xmax - xmin, ymax - ymin])

    image = rotate_image(image, angle)
    return image, new_keypoints, new_bbox


def image_random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:    # h,w,c
        img = img[::-1, :,:]
    if x_flip:
        img = img[:, ::-1,:]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_point(point, size, y_flip=False, x_flip=False):
    """Modify points according to image flips.

    Args:
        point (~numpy.ndarray or list of arrays): See the table below.
        size (tuple): A tuple of length 2. The height and the width
            of the image, which is associated with the points.
        y_flip (bool): Modify points according to a vertical flip of
            an image.
        x_flip (bool): Modify keypoipoints according to a horizontal flip of
            an image.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`point`, ":math:`(R, K, 2)` or :math:`[(K, 2)]`", \
        :obj:`float32`, ":math:`(y, x)`"

    Returns:
        ~numpy.ndarray or list of arrays:
        Points modified according to image flips.

    """
    H, W = size
    if isinstance(point, np.ndarray):
        out_point = point.copy()
        if y_flip:
            out_point[:,0] = H - out_point[:,0]
        if x_flip:
            out_point[:,1] = W - out_point[:,1]
    else:
        out_point = []
        for pnt in point:
            pnt = pnt.copy()
            if y_flip:
                pnt[:, 0] = H - pnt[:, 0]
            if x_flip:
                pnt[:, 1] = W - pnt[:, 1]
            out_point.append(pnt)
    return out_point


def random_flip(image, keypoints, bbox, is_labeled, is_visible, flip_indices):
    """
    random x_flip
    Note that if image is flipped, `flip_indices` translate elements.
    e.g. left_shoulder -> right_shoulder.
    """
    H, W,_ = image.shape
    image, param = image_random_flip(image, x_random=True, return_param=True)

    if param['x_flip']:
        keypoints = [
            flip_point(points, (H, W), x_flip=True)[flip_indices]
            for points in keypoints
        ]

        is_labeled = [label[flip_indices] for label in is_labeled]
        is_visible = [vis[flip_indices] for vis in is_visible]

        new_bbox = []
        for x, y, w, h in bbox:
            [[y, x]] = flip_point(np.array([[y, x + w]]), (H, W), x_flip=True)
            new_bbox.append([x, y, w, h])
        bbox = new_bbox

    return image, keypoints, bbox, is_labeled, is_visible


def random_resize(image, keypoints, bbox):
    # Random resize
    H, W,_ = image.shape
    scalew, scaleh = np.random.uniform(0.7, 1.5, 2)
    resizeW, resizeH = int(W * scalew), int(H * scaleh)
    image, keypoints, bbox = resize(image, keypoints, bbox, (resizeH, resizeW))
    return image, keypoints, bbox


def random_crop(image, keypoints, bbox, is_labeled):

    crop_target = np.random.choice(
            ['random_sized_crop', 'crop_all_humans','none'],
            p=[0.1,0.4 ,0.5],
        )
    if crop_target == 'random_sized_crop':
        image, keypoints, bbox= random_resize(image, keypoints, bbox)
        image, keypoints, bbox = random_sized_crop(image, keypoints, bbox)
    elif crop_target == 'crop_all_humans':
        image, keypoints, bbox= crop_all_humans(image, keypoints, bbox, is_labeled)
    else:
        pass
    return image, keypoints, bbox


def pad_keypoint(point,h_pad,w_pad):

    if isinstance(point, np.ndarray):
        out_point = point.copy()
        out_point[:,  0] = h_pad + point[:, 0]
        out_point[:, 1] = w_pad + point[:, 1]
        # out_point = out_point.tolist()
    else:
        out_point = []
        for pnt in point:
            out_pnt = pnt.copy()
            out_pnt[:, 0] = h_pad + pnt[:, 0]
            out_pnt[:, 1] = w_pad + pnt[:, 1]
            out_point.append(out_pnt)
    return out_point


def Pad(image,keypoints,bbox):
    H,W,_ = image.shape
    #padsize = (max(H,W) -min(H,W))//2
    h_pad = int(np.clip(((max(H,W) - H) + 1) // 2, 0, 1e6))  # 填充在两边
    w_pad = int(np.clip(((max(H,W) - W) + 1) // 2, 0, 1e6))
    pad = ((h_pad, h_pad), (w_pad, w_pad))
    image = np.stack([np.pad(image[:, :, c], pad,
                             mode='constant',
                             constant_values=[123, 116, 103][c]) for c in range(3)], axis=2)

    keypoints = [
        pad_keypoint(points, h_pad,w_pad)
        for points in keypoints
    ]

    new_bbox = []
    for x, y, bw, bh in bbox:
        [[y, x]] = pad_keypoint(np.array([[y, x]]), h_pad, w_pad)
        new_bbox.append(np.array([x, y, bw, bh]))

    return image,keypoints,new_bbox








if __name__=="__main__":

    pass

