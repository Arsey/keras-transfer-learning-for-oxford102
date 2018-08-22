from glob import glob
from shutil import rmtree
import os
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField
from wtforms.validators import DataRequired, ValidationError, NumberRange
import numpy as np
import PIL
from PIL import Image
import scipy.misc

INPUT_DATA_DIR = '/input-data'
DATASETS_DIR = '/datasets'

ALLOWED_IMGS_EXTENSIONS = ['*.jpeg', '*.jpg', '*.png', '*.tif']


class TextNumber(object):
    def __init__(self, min=0, max=99, message=None):
        self.min = int(min)
        self.max = int(max)
        if not message:
            message = u'Field must be between %i and %i' % (min, max)
        self.message = message

    def __call__(self, form, field):
        l = field.data and len(field.data) or 0
        if l < self.min or self.max != -1 and l > self.max:
            raise ValidationError(self.message)


class NewDatasetFrom(FlaskForm):
    train_folder = StringField('train_folder', validators=[DataRequired()])

    def validate_train_folder(self, field):
        train_folder = os.path.join(INPUT_DATA_DIR, field.data)

        msg = None
        if not os.path.exists(train_folder):
            available_dirs = os.listdir(INPUT_DATA_DIR)
            msg = 'Train folder does not exist. Available dirs are: <br/> {}'.format('<br/>'.join(available_dirs))
        elif not os.path.isdir(train_folder):
            msg = 'Train folder is not a directory'
        elif not len(os.listdir(train_folder)):
            msg = 'Train folder must be a folder which holds subfolders full of images. ' \
                  'Each subfolder should be named according to the desired label for the images that it holds.'

        if msg:
            raise ValidationError(msg)

    dataset_name = StringField('dataset_name', validators=[DataRequired()])

    def validate_dataset_name(self, field):
        dataset_dir = os.path.join(DATASETS_DIR, field.data)

        msg = None
        # TODO: uncomment
        # if os.path.exists(dataset_dir):
        #     msg = 'There is already a dataset with the "{}" name. ' \
        #           'Please try with different dataset name or delete the existing one.'.format(field.data)

        if msg:
            raise ValidationError(msg)

    resize_channels = SelectField(
        'resize_channels',
        choices=[('3', 'Color'), ('1', 'Grayscale')],
        default='3'
    )

    resize_width = StringField('resize_width', validators=[DataRequired(), TextNumber(min=2, max=8192)], default=256)
    resize_height = StringField('resize_height', validators=[DataRequired(), TextNumber(min=2, max=8192)], default=256)

    resize_mode = SelectField(
        'resize_mode',
        choices=[('crop', 'Crop'), ('squash', 'Squash'), ('fill', 'Fill'), ('half_crop', 'Half Crop')],
        default='squash'
    )

    folder_train_min_per_class = StringField('folder_train_min_per_class', validators=[TextNumber(min=0)], default=2)
    folder_train_max_per_class = StringField('folder_train_max_per_class', default=None)

    folder_pct_val = StringField('folder_pct_val', validators=[TextNumber(min=0, max=99)], default=25)
    folder_pct_test = StringField('folder_pct_test', validators=[TextNumber(min=0, max=99)], default=0)

    encoding = SelectField(
        'encoding',
        choices=[('jpg', 'JPEG (lossy, 90% quality)'), ('png', 'PNG')],
        default='png'
    )


def create_dataset_dirs(dataset_dir, classes, with_test=True):
    # TODO: remove this line and delete datasets separately. for now just overwirte
    rmtree(dataset_dir, ignore_errors=True)

    split_dirs = ['train', 'valid']
    if with_test:
        split_dirs.append('test')

    for split in split_dirs:
        for cls in classes:
            os.makedirs(os.path.join(dataset_dir, split, cls))


def get_all_allowed_images_from_dir(dirname):
    images = []
    for ext in ALLOWED_IMGS_EXTENSIONS:
        images.extend(glob(os.path.join(dirname, ext)))
    return images


def get_splits(images_list, valid, test):
    train = 100 - (valid + test)
    n = len(images_list)

    train_n = int(n * (train / 100.))
    train_split = images_list[:train_n]

    valid_n = int(n * (valid / 100.))
    valid_split = images_list[train_n:train_n + valid_n]

    test_split = images_list[train_n + valid_n:]

    assert len(train_split) + len(valid_split) + len(test_split) == len(images_list)

    return train_split, valid_split, test_split


def image_to_array(image, channels=None):
    """
    Returns an image as a np.array
    Arguments:
    image -- a PIL.Image or numpy.ndarray
    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    """

    if channels not in [None, 1, 3, 4]:
        raise ValueError('unsupported number of channels: %s' % channels)

    if isinstance(image, PIL.Image.Image):
        # Convert image mode (channels)
        if channels is None:
            image_mode = image.mode
            if image_mode not in ['L', 'RGB', 'RGBA']:
                raise ValueError('unknown image mode "%s"' % image_mode)
        elif channels == 1:
            # 8-bit pixels, black and white
            image_mode = 'L'
        elif channels == 3:
            # 3x8-bit pixels, true color
            image_mode = 'RGB'
        elif channels == 4:
            # 4x8-bit pixels, true color with alpha
            image_mode = 'RGBA'
        if image.mode != image_mode:
            image = image.convert(image_mode)
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.reshape(image.shape[:2])
        if channels is None:
            if not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in [3, 4])):
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 1:
            if image.ndim != 2:
                if image.ndim == 3 and image.shape[2] in [3, 4]:
                    # color to grayscale. throw away alpha
                    image = np.dot(image[:, :, :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                else:
                    raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 3:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 3).reshape(image.shape + (3,))
            elif image.shape[2] == 4:
                # throw away alpha
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError('invalid image shape: %s' % (image.shape,))
        elif channels == 4:
            if image.ndim == 2:
                # grayscale to color
                image = np.repeat(image, 4).reshape(image.shape + (4,))
                image[:, :, 3] = 255
            elif image.shape[2] == 3:
                # add alpha
                image = np.append(image, np.zeros(image.shape[:2] + (1,), dtype='uint8'), axis=2)
                image[:, :, 3] = 255
            elif image.shape[2] != 4:
                raise ValueError('invalid image shape: %s' % (image.shape,))
    else:
        raise ValueError('resize_image() expected a PIL.Image.Image or a numpy.ndarray')

    return image


def get_padding(current_size, target_size, padding_type):
    diff = current_size - target_size
    if padding_type == 'both':
        return diff / 2
    return diff


def get_noise(noise_size, channels, resize_mode, fill_color):
    if channels > 1:
        noise_size += (channels,)

    if resize_mode != 'fill_one_color':
        return np.random.randint(0, 255, noise_size).astype('uint8')

    return np.ones(shape=noise_size).astype('uint8') * fill_color


def concatenate_image_with_noise(img, noise, padding_type, axis):
    if padding_type == 'first':
        return np.concatenate((noise, img), axis=axis)
    elif padding_type == 'last':
        return np.concatenate((img, noise), axis=axis)
    else:
        return np.concatenate((noise, img, noise), axis=axis)


def resize_image(image, height, width, channels=None, resize_mode=None, fill_color=0, padding_type='both'):
    """
    Resizes an image and returns it as a np.array
    Arguments:
    image -- a PIL.Image or numpy.ndarray
    height -- height of new image
    width -- width of new image
    Keyword Arguments:
    channels -- channels of new image (stays unchanged if not specified)
    resize_mode -- can be crop, squash, fill or half_crop
    padding_type -- can be both, first, last
    """
    padding = 0

    if resize_mode is None:
        resize_mode = 'squash'
    if resize_mode not in ['crop', 'squash', 'fill', 'fill_one_color', 'half_crop']:
        raise ValueError('resize_mode "%s" not supported' % resize_mode)

    # convert to array
    image = image_to_array(image, channels)

    # No need to resize
    if image.shape[0] == height and image.shape[1] == width:
        return image, padding

    # Resize
    interp = 'bilinear'

    width_ratio = float(image.shape[1]) / width
    height_ratio = float(image.shape[0]) / height

    if resize_mode == 'squash' or width_ratio == height_ratio:
        return scipy.misc.imresize(image, (height, width), interp=interp), padding
    elif resize_mode == 'crop':
        # resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio:
            resize_height = height
            resize_width = int(round(image.shape[1] / height_ratio))
        else:
            resize_width = width
            resize_height = int(round(image.shape[0] / width_ratio))
        image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)

        # chop off ends of dimension that is still too long
        if width_ratio > height_ratio:
            start = int(round((resize_width - width) / 2.0))
            return image[:, start:start + width], padding
        else:
            start = int(round((resize_height - height) / 2.0))
            return image[start:start + height, :], padding
    else:
        if resize_mode == 'fill' or resize_mode == 'fill_one_color':
            # resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio:
                resize_width = width
                resize_height = int(round(image.shape[0] / width_ratio))
                if (height - resize_height) % 2 == 1:
                    resize_height += 1
            else:
                resize_height = height
                resize_width = int(round(image.shape[1] / height_ratio))
                if (width - resize_width) % 2 == 1:
                    resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
        elif resize_mode == 'half_crop':
            # resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = int(round(image.shape[1] / new_ratio))
            resize_height = int(round(image.shape[0] / new_ratio))
            if width_ratio > height_ratio and (height - resize_height) % 2 == 1:
                resize_height += 1
            elif width_ratio < height_ratio and (width - resize_width) % 2 == 1:
                resize_width += 1
            image = scipy.misc.imresize(image, (resize_height, resize_width), interp=interp)
            # chop off ends of dimension that is still too long
            if width_ratio > height_ratio:
                start = int(round((resize_width - width) / 2.0))
                image = image[:, start:start + width]
            else:
                start = int(round((resize_height - height) / 2.0))
                image = image[start:start + height, :]
        else:
            raise Exception('unrecognized resize_mode "%s"' % resize_mode)

        # fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio:
            padding = get_padding(height, resize_height, padding_type)
            noise = get_noise((padding, width), channels, resize_mode, fill_color)
            image = concatenate_image_with_noise(image, noise, padding_type, 0)
        else:
            padding = get_padding(width, resize_width, padding_type)
            noise = get_noise((height, padding), channels, resize_mode, fill_color)
            image = concatenate_image_with_noise(image, noise, padding_type, 1)

        return image, padding


def process_images(images_list, dst_dir, resize_channels=3, resize_width=256, resize_height=256, resize_mode='squash',
                   encoding='png', ):
    for i in images_list:
        img = Image.open(i)
        img = resize_image(
            img,
            int(resize_height),
            int(resize_width),
            channels=int(resize_channels),
            resize_mode=resize_mode,
            padding_type='both')
        Image.fromarray(img[0]).save(os.path.join(dst_dir, os.path.basename(i[:-3] + encoding)))


def create(
        train_folder,
        dataset_name,
        resize_channels,
        resize_width,
        resize_height,
        resize_mode,
        encoding,
        folder_train_min_per_class=2,
        folder_train_max_per_class=None,
        folder_pct_val=25,
        folder_pct_test=0
):
    classes_dirs = glob(os.path.join(INPUT_DATA_DIR, train_folder, '**'))
    classes = [os.path.basename(i) for i in classes_dirs]

    # init dataset dirs
    dataset_dir = os.path.join(DATASETS_DIR, dataset_name)
    create_dataset_dirs(dataset_dir, classes)

    for class_dir in classes_dirs:
        class_images = get_all_allowed_images_from_dir(class_dir)
        if len(class_images):
            train, valid, test = get_splits(
                images_list=class_images,
                valid=int(folder_pct_val),
                test=int(folder_pct_test)
            )

            train_dir = os.path.join(dataset_dir, 'train', os.path.basename(class_dir))
            valid_dir = os.path.join(dataset_dir, 'valid', os.path.basename(class_dir))
            test_dir = os.path.join(dataset_dir, 'test', os.path.basename(class_dir))

            process_images(train, train_dir, resize_channels, resize_width, resize_height, resize_mode, encoding)
            process_images(valid, valid_dir, resize_channels, resize_width, resize_height, resize_mode, encoding)
            process_images(test, test_dir, resize_channels, resize_width, resize_height, resize_mode, encoding)


def delete(name):
    dataset_path = os.path.join(DATASETS_DIR, name)
    if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
        rmtree(dataset_path, ignore_errors=True)


def get_datasets_list():
    datasets = glob(os.path.join(DATASETS_DIR, '**'))
    return [os.path.basename(i) for i in datasets]
