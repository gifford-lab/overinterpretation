"""Runs backward selection on a single CIFAR image for a pre-trained model.

Unlike SIS, only runs backward selection once. In the returned SISResult object,
ignore the SIS and just use backward selection values.

Example usage:
python run_sis_on_adv_robust.py \
  --model_checkpoint_dir=./madrynet/models/adv_trained \
  --image_idx=10 \
  --out_dir=./madrynet/sis_results \
  --batch_size=128 \
  --sis_threshold=0.99
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import json
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from sufficient_input_subsets import sis
from madrynet.cifar10_challenge.model import Model


FLAGS = flags.FLAGS

flags.DEFINE_float('sis_threshold', 0, 'Threshold to use for SIS.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for model inference.')
flags.DEFINE_integer('image_idx', None, 'Image index (into CIFAR) test set.')
# flags.DEFINE_integer('gpu', 0, 'GPU (for cuda_visible_devices).')
flags.DEFINE_string('out_dir', None, 'Path to write out file with SIS.')
flags.DEFINE_string(
    'model_checkpoint_dir', None, 'Path to model checkpoint directory.')

__TF_SESSION__ = None


def tf_config():
    """Configures TensorFlow and returns corresponding tf.Session object."""
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    return sess


def make_f_cnn(sess, output_tensor, input_tensor, class_idx, batch_size=128):
    def f_cnn(batch_of_inputs):
        preds = predict(
            sess, output_tensor, input_tensor, batch_of_inputs,
            batch_size=batch_size)
        return preds[:, class_idx]
    return f_cnn


def predict(sess, output_tensor, input_tensor, x, batch_size=128):
    x = np.array(x)
    preds = []
    for batch_idx in range(int(np.ceil(x.shape[0] / batch_size))):
        x_batch = x[batch_size*batch_idx:batch_size*(batch_idx+1)]
        batch_preds = sess.run(
            [output_tensor], feed_dict={input_tensor: x_batch})[0]
        preds.append(batch_preds)
    preds = np.vstack(preds)
    assert preds.shape[0] == x.shape[0]
    return preds


def sis_result_to_dict(sis_result):
    return {
        'sis': sis_result.sis.tolist(),
        'ordering_over_entire_backselect': sis_result.ordering_over_entire_backselect.tolist(),
        'values_over_entire_backselect': sis_result.values_over_entire_backselect.tolist(),
        'mask': sis_result.mask.tolist(),
    }


def create_output_dict(collection, sis_threshold, model_checkpoint_dir,
                       image_idx, target_class_idx):
    return {
        'collection': [sis_result_to_dict(sr) for sr in collection],
        'sis_threshold': sis_threshold,
        'model_checkpoint_dir': model_checkpoint_dir,
        'image_idx': image_idx,
        'target_class_idx': target_class_idx,
    }


def write_dict_to_json(dict_to_write, filepath):
    with open(filepath, 'w') as f:
        json.dump(dict_to_write, f)


def main(argv):
    del argv

    global __TF_SESSION__
    __TF_SESSION__ = tf_config()  # cuda_visible_devices=str(FLAGS.gpu))
    sess = __TF_SESSION__

    logging.basicConfig(level=logging.INFO)

    sis_threshold = FLAGS.sis_threshold
    batch_size = FLAGS.batch_size
    model_checkpoint_dir = FLAGS.model_checkpoint_dir
    out_dir = FLAGS.out_dir
    image_idx = FLAGS.image_idx

    logging.info('SIS threshold: %f' % sis_threshold)
    logging.info('Batch size: %d' % batch_size)
    logging.info('Model checkpoint dir: %s' % model_checkpoint_dir)
    logging.info('Out dir: %s' % out_dir)
    logging.info('Image idx: %s' % image_idx)

    out_path = os.path.join(out_dir, 'test_%d_sis.json' % image_idx)
    logging.info('Will write to outpath: %s' % out_path)

    # Check if outfile already exists.
    if os.path.exists(out_path):
        logging.info('Outfile already exists. Exiting.')
        return

    # Load model.
    model = Model(mode='eval')
    model_softmax = tf.nn.softmax(model.pre_softmax)
    input_tensor = model.x_input
    checkpoint = tf.train.latest_checkpoint(model_checkpoint_dir)
    # Restore the checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    logging.info('Loaded TF model.')

    # Load and preprocess CIFAR data.
    logging.info('Loading CIFAR data.')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    X_TRAIN_MEAN = np.array([125.3, 123.0, 113.9]) / 255.
    X_TRAIN_STD = np.array([63.0, 62.1, 66.7]) / 255.

    x_train = (x_train - X_TRAIN_MEAN) / X_TRAIN_STD
    x_test = (x_test - X_TRAIN_MEAN) / X_TRAIN_STD

    # Define fully masked input.
    fully_masked_input = np.zeros((32, 32, 3), dtype='float32')

    # Run SIS.
    original_image = x_test[image_idx]
    initial_prediction = predict(
        sess, model_softmax, input_tensor, np.array([original_image]))[0]
    target_class_idx = int(np.argmax(initial_prediction))
    logging.info('Target class idx:  %d' % target_class_idx)
    f_class = make_f_cnn(
        sess, model_softmax, input_tensor, target_class_idx,
        batch_size=batch_size)
    logging.info('Starting to run SIS.')
    initial_mask = sis.make_empty_boolean_mask_broadcast_over_axis(
        original_image.shape, 2)
    sis_result = sis.find_sis(
        f_class,
        sis_threshold,
        original_image,
        initial_mask,
        fully_masked_input,
    )
    collection = [sis_result]
    logging.info('Done running SIS.')

    # Write SIS collection to file.
    output_dict = create_output_dict(
        collection, sis_threshold, model_checkpoint_dir, image_idx,
        target_class_idx)
    logging.info('Writing SIS output to: %s' % out_path)
    # util.create_directory(out_dir)
    write_dict_to_json(output_dict, out_path)


if __name__ == '__main__':
    flags.mark_flag_as_required('model_checkpoint_dir')
    flags.mark_flag_as_required('out_dir')
    flags.mark_flag_as_required('image_idx')
    app.run(main)
