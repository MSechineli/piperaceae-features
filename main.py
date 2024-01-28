import glob
import math
import numpy as np
import os
import pandas as pd
import pathlib
import tensorflow as tf
import logging
import time
import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=datetime.datetime.now().strftime('time_%H_%M_%d_%m_%Y.log'), encoding='utf-8', level=logging.INFO)

from PIL import Image, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATHBASE = '/home/msechineli/Documents/tcc'
PATCHES = [3]


def next_patch(spec, n):
    step = math.floor(spec.shape[0] / n)
    for i in range(n):
        yield spec[i * step:(i + 1) * step, :, :]


def get_model(model, **kwargs):
    if model == 'xception':
        return tf.keras.applications.xception.Xception(**kwargs), tf.keras.applications.xception.preprocess_input
    if model == 'DenseNet201':
        return tf.keras.applications.densenet.DenseNet201(**kwargs), tf.keras.applications.densenet.preprocess_input
    if model == 'vgg19':
        return tf.keras.applications.vgg19.VGG19(**kwargs), tf.keras.applications.vgg19.preprocess_input
    if model == 'vgg16':
        return tf.keras.applications.vgg16.VGG16(**kwargs), tf.keras.applications.vgg16.preprocess_input
    if model == 'resnet50v2':
        return tf.keras.applications.resnet_v2.ResNet50V2(**kwargs), tf.keras.applications.resnet_v2.preprocess_input
    if model == 'mobilenetv2':
        return tf.keras.applications.mobilenet_v2.MobileNetV2(
            **kwargs), tf.keras.applications.mobilenet_v2.preprocess_input

    raise ValueError


def adjust_contrast(contrast, image):
    enhancer = ImageEnhance.Contrast(image)
    im_contrast = enhancer.enhance(contrast)
    return im_contrast


def extract_features(cnn, color, contrast, dataset, gpuid, folds, image_size, input_path, level, minimum_image, output_path, patches, region):
    logging.info("GPU ID: %s" % str(gpuid))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    spec_height = image_size[0]
    spec_width = image_size[1]
    input_path_proto = os.path.join(input_path, 'f%d', '*.jpeg')

    for n_patches in patches:
        logging.info('Slicing images into %d non-overlapping patches...' % (n_patches))
        tf.keras.backend.clear_session()

        input_shape = (math.floor(spec_height / n_patches), spec_width, 3)
        model, preprocess_input = get_model(cnn, weights='imagenet', include_top=False,
                                            input_shape=input_shape, pooling='avg')
        total_samples = 0
        n_features = 0
        for fold in folds:
            logging.info('Extracting features for fold %d...' % (fold))
            if len(glob.glob(input_path_proto % (fold))) == 0:
                raise RuntimeError('No files found in: %s' % (input_path_proto % (fold)))

            features = []
            for fname in sorted(glob.glob(input_path_proto % (fold))):
                logging.info('fname: %s' % fname)
                im_sliced = []
                im = tf.keras.preprocessing.image.load_img(fname)
                im_contrast = adjust_contrast(contrast, im)
                spec = tf.keras.preprocessing.image.img_to_array(im_contrast)
                for p in next_patch(spec, n_patches):
                    p = preprocess_input(p)
                    im_sliced.append(tf.keras.preprocessing.image.array_to_img(p))
                    p = np.expand_dims(p, axis=0)
                    features.append(model.predict(p))

                save_image(contrast, dataset, fname, im_sliced)

            features = np.concatenate(features)

            for format in ['npy', 'npz']:
                save_file(format, features, fold, n_patches, output_path)

            n_samples, n_features = features.shape

            total_samples += n_samples
        save_information(color, cnn, contrast, dataset, image_size, input_path, level, minimum_image, n_features, output_path, n_patches, region, total_samples)


def save_image(contrast, dataset, fname, im_sliced):
    dir_fname = str(pathlib.Path(fname).stem)
    dir = str(pathlib.Path(fname).parent).replace(dataset, '%s_CONTRAST_%s' % (dataset, contrast))
    dir = os.path.join(dir, dir_fname)
    fname = str(pathlib.Path(fname).name)

    if not os.path.exists(dir):
        os.makedirs(dir)

    for i, im in enumerate(im_sliced, start=1):
        fname_sliced = os.path.join(dir, '%s_%s' % (i, fname))

        if not os.path.isfile(fname_sliced):
            logging.info('%s saved' % fname_sliced)
            tf.keras.preprocessing.image.save_img(fname_sliced, im)


def save_file(extension, features, fold, n_patches, output_path):
    output_path = os.path.join(output_path, extension)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, 'fold-%d_patches-%d.' + extension)
    output_filename = output_path % (fold, n_patches)
    logging.info('%s save' % output_filename)
    if extension == 'npy':
        np.save(output_filename, features, allow_pickle=True)
    else:
        np.savez_compressed(output_filename, x=features, y=np.repeat(fold, features.shape[0]))


def save_information(color, cnn, contrast, dataset, image_size, input_path, level, minimum_image, n_features, output_path, patch, region, total_samples):
    height = str(image_size[0])
    width = str(image_size[1])
    index = ['cnn', 'color', 'contrast', 'dataset', 'height', 'width', 'level', 'minimum_image', 'input_path', 'output_path',
             'patch', 'n_features', 'total_samples']
    data = [cnn, color, contrast, dataset, height, width, level, minimum_image, input_path, output_path, patch, n_features, total_samples]

    if region:
        index.append('region')
        data.append(region)

    df = pd.DataFrame(data, index=index)
    filename = os.path.join(output_path, 'info.csv')
    logging.info('%s saved' % filename)
    df.to_csv(filename, sep=';', index=index, header=None, lineterminator='\n', doublequote=True)


def prepare(cnn, color, contrast, dataset, image_size, level, minimum_image, input_path, output_path, region=None):
    if not os.path.exists(input_path):
        raise SystemError('path (%s) not exists' % input_path)

    list_folders = [p for p in pathlib.Path(input_path).glob('*') if p.is_dir()]
    folds = len(list_folders)
    folds = list(range(1, folds + 1))
    image_size = (int(image_size), int(image_size))
    gpuid = 0
    patches = PATCHES

    path_features = output_path.replace(dataset, '%s_features_CONTRAST_%s' % (dataset, contrast))

    if not os.path.exists(path_features):
        os.makedirs(path_features)

    logging.info('Feature Extraction Parameters')
    logging.info('Pre-trained model: %s' % cnn)
    logging.info('Non-overlapping patches per image: %s' % str(patches))
    logging.info('Folds: %s' % str(folds))
    logging.info('Image Dimensions h=%s, w=%s ' % (image_size, image_size))
    logging.info('Format string for input: %s ' % input_path)
    logging.info('Format string for output: %s ' % output_path)
    logging.info('GPU ID: %d' % gpuid)

    initTime = time.time()
    extract_features(cnn, color, contrast, dataset, gpuid, folds, image_size, input_path, level, minimum_image, path_features, patches, region)
    endTime = time.time()
    logging.info("Time to extract_features %f", (endTime - initTime))


def main():
    for contrast in [1.2]:
        for dataset in ['pr_dataset']:
            for cnn in ['vgg19']:
                for color in ['GRAYSCALE']:
                    for image_size in ['512']:
                        for minimum_image in ['20', '10', '5']:
                            for level in ['specific_epithet_trusted']:
                                    logging.info('cnn: %s color: %s dataset: %s image_size: %s level: %s minimum_image: %s '
                                          % (cnn, color, dataset, image_size, level, minimum_image))
                                    if 'regions_dataset' == dataset:
                                        for region in ['Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste']:
                                            path = os.path.join(PATHBASE, dataset, color, level, region, image_size,
                                                                minimum_image)
                                            output_path = os.path.join(PATHBASE, dataset, color, level, image_size, region, minimum_image, cnn)
                                            prepare(cnn, color, contrast, dataset, image_size, level, minimum_image, path, output_path, region=region)
                                    else:
                                        path = os.path.join(PATHBASE, dataset, color, level, image_size, minimum_image)
                                        output_path = os.path.join(PATHBASE, dataset, color, level, image_size,
                                                                   minimum_image, cnn)
                                        prepare(cnn, color, contrast, dataset, image_size, level, minimum_image, path, output_path)
                                



if __name__ == '__main__':
    main()
