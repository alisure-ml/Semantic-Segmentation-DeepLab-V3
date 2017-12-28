import os
import tensorflow as tf
import time

from config import *
from libs.datasets.dataset_factory import read_data
from libs.nets import deeplabv3

import tensorflow.contrib.slim as slim
import tensorflow.contrib.metrics as tcm


def save(saver, sess, logdir, step):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, os.path.join(logdir, "model.ckpt"), global_step=step)
    print('The checkpoint has been created.')
    pass


def load(saver, sess, ckpt_dir):
    if args.ckpt == 0:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        ckpt_path = ckpt.model_checkpoint_path
    else:
        ckpt_path = ckpt_dir + '/model.ckpt-%i' % args.ckpt
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
    pass


def main():

    image_batch, label_batch = read_data(is_training=True)

    # Create network.
    net, end_points = deeplabv3(image_batch, num_classes=args.num_classes, depth=args.num_layers, is_training=True)

    # For a small batch size, it is better to keep
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) 
    # and beta (offset)
    # if they are presented in var_list of the optimizer definition.

    # Predictions.
    raw_output = end_points['resnet{}/logits'.format(args.num_layers)]
    # Which variables to load. Running means and variances are not trainable, thus all_variables() should be restored.
    restore_var = [v for v in tf.global_variables() if 'fc' not in v.name or not args.not_restore_last]
    if args.freeze_bn:
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    else:
        all_trainable = [v for v in tf.trainable_variables()]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name]

    # Upsample the logits instead of donwsample the ground truth
    raw_output_up = tf.image.resize_bilinear(raw_output, [args.input_size, args.input_size])

    # Predictions: ignoring all predictions with labels greater or equal than n_classes
    label_proc = tf.squeeze(label_batch)
    mask = label_proc <= args.num_classes
    seg_logits = tf.boolean_mask(raw_output_up, mask)
    seg_gt = tf.cast(tf.boolean_mask(label_proc, mask), tf.int32)

    # Pixel-wise softmax loss.
    seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits, labels=seg_gt))
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tot_loss = seg_loss + reg_loss

    seg_pred = tf.argmax(seg_logits, axis=1)
    train_mean_iou, train_update_mean_iou = tcm.streaming_mean_iou(seg_pred, seg_gt, args.num_classes, name="train_iou")
    # 初始化iou,重新计算
    train_initializer = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_iou"))

    # Define loss and optimisation parameters.
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    # learning_rate = base_lr
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))

    seg_loss_sum = tf.summary.scalar('loss/seg', seg_loss)
    reg_loss_sum = tf.summary.scalar('loss/reg', reg_loss)
    tot_loss_sum = tf.summary.scalar('loss/tot', tot_loss)
    train_iou_sum = tf.summary.scalar('accuracy/train_mean_iou', train_mean_iou)
    lr_sum = tf.summary.scalar('params/learning_rate', learning_rate)
    train_sum_op = tf.summary.merge([seg_loss_sum, reg_loss_sum, tot_loss_sum, train_iou_sum, lr_sum])

    # 下面是 VAL
    image_batch_val, label_batch_val = read_data(is_training=False)
    _, end_points_val = deeplabv3(image_batch_val, num_classes=args.num_classes,
                                  depth=args.num_layers, reuse=True, is_training=False)
    raw_output_val = end_points_val['resnet{}/logits'.format(args.num_layers)]
    nh, nw = tf.shape(image_batch_val)[1], tf.shape(image_batch_val)[2]

    seg_logits_val = tf.image.resize_bilinear(raw_output_val, [nh, nw])
    seg_pred_val = tf.reshape(tf.expand_dims(tf.argmax(seg_logits_val, axis=3), 3), [-1, ])
    seg_gt_val = tf.reshape(tf.cast(label_batch_val, tf.int32), [-1, ])

    mask_val = seg_gt_val <= args.num_classes - 1

    seg_pred_val = tf.boolean_mask(seg_pred_val, mask_val)
    seg_gt_val = tf.boolean_mask(seg_gt_val, mask_val)

    val_mean_iou, val_update_mean_iou = tcm.streaming_mean_iou(seg_pred_val, seg_gt_val, num_classes=args.num_classes, name="val_iou")
    # 初始化iou,重新计算
    val_initializer = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="val_iou"))

    val_iou_sum = tf.summary.scalar('accuracy/val_mean_iou', val_mean_iou)
    test_sum_op = tf.summary.merge([val_iou_sum])

    global_step = tf.train.get_or_create_global_step()

    opt = tf.train.MomentumOptimizer(learning_rate, args.momentum)

    # grads_conv = tf.gradients(tot_loss, conv_trainable)
    # train_op = opt.apply_gradients(zip(grads_conv, conv_trainable))
    train_op = slim.learning.create_train_op(tot_loss, opt, global_step=global_step,
                                             variables_to_train=conv_trainable, summarize_gradients=True)

    # Set up tf session and initialize variables.
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)

        # Load variables if the checkpoint is provided.
        if args.ckpt > 0 or args.restore_from is not None:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, args.snapshot_dir)

        tf.set_random_seed(args.random_seed)
        coord = tf.train.Coordinator()  # Create queue coordinator.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)  # Start queue threads.

        # tf.get_default_graph().finalize()
        summary_writer = tf.summary.FileWriter(args.snapshot_dir, sess.graph)

        # Iterate over training steps.
        for step in range(args.ckpt, args.num_steps):
            start_time = time.time()
            tot_loss_float, _, lr_float, _, train_summary = sess.run(
                [tot_loss, train_op, learning_rate, train_update_mean_iou, train_sum_op], feed_dict={step_ph: step})
            train_mean_iou_float = sess.run(train_mean_iou)
            duration = time.time() - start_time
            print('step {:d}, tot_loss = {:.6f}, mean_iou = {:.6f}, lr: {:.6f}({:.3f} sec/step)'
                  .format(step, tot_loss_float, train_mean_iou_float, lr_float, duration))

            if step % args.save_pred_every == 0 and step > args.ckpt:
                summary_writer.add_summary(train_summary, step)
                sess.run(val_initializer)

                test_summary = None
                for val_step in range(NUM_VAL - 1):
                    _, test_summary = sess.run([val_update_mean_iou, test_sum_op], feed_dict={step_ph: step})

                summary_writer.add_summary(test_summary, step)
                val_mean_iou_float = sess.run(val_mean_iou)

                save(saver, sess, args.snapshot_dir, step)
                print('step {:d}, train_mean_iou: {:.6f}, val_mean_iou: {:.6f}'
                      .format(step, train_mean_iou_float, val_mean_iou_float))
                sess.run(train_initializer)
            pass

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
