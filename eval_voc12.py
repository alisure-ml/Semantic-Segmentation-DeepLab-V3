import tensorflow as tf
import time
import os

from config import *
from libs.datasets.dataset_factory import read_data
from libs.nets import deeplabv3
import tensorflow.contrib.metrics as tcm


def save(saver, sess, logdir, step):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, os.path.join(logdir, "model.ckpt"), global_step=step)
    print('The checkpoint has been created.')
    pass


def load(saver, sess, ckpt_dir):
    if args.ckpt == 0:
        ckpt_path = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
    else:
        ckpt_path = ckpt_dir + '/model.ckpt-%i' % args.ckpt
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
    pass


def main():
    image_batch, label_batch = read_data(is_training=args.is_training)

    # Create network.
    net, end_points = deeplabv3(image_batch, num_classes=args.num_classes, depth=args.num_layers, is_training=False)

    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Which variables to load. Running means and variances are not trainable, thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]

    # Predictions.
    raw_output = end_points['resnet{}/logits'.format(args.num_layers)]
    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    nh, nw = tf.shape(image_batch)[1], tf.shape(image_batch)[2]
    seg_logits = tf.image.resize_bilinear(raw_output, [nh, nw])

    seg_pred = tf.reshape(tf.expand_dims(tf.argmax(seg_logits, axis=3), 3), [-1, ])
    seg_gt = tf.reshape(tf.cast(label_batch, tf.int32), [-1, ])

    mask = seg_gt <= args.num_classes - 1

    seg_pred = tf.boolean_mask(seg_pred, mask)
    seg_gt = tf.boolean_mask(seg_gt, mask)

    mean_iou, update_mean_iou = tcm.streaming_mean_iou(seg_pred, seg_gt, num_classes=args.num_classes)

    # Set up tf session and initialize variables.
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])

        # Load variables if the checkpoint is provided.
        if args.ckpt > 0 or args.restore_from is not None:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, args.snapshot_dir)

        tf.set_random_seed(args.random_seed)
        coord = tf.train.Coordinator()  # Create queue coordinator.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # Start queue threads.

        tf.get_default_graph().finalize()

        for step in range(1449):
            start_time = time.time()
            mean_iou_float, _ = sess.run([mean_iou, update_mean_iou])
            duration = time.time() - start_time
            print('step {:d}, mean_iou: {:.6f}({:.3f} sec/step)'.format(step, mean_iou_float, duration))

        coord.request_stop()
        coord.join(threads)
        pass
    pass


if __name__ == '__main__':
    main()
