import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import DTA_model as model
# from tensorflow.python.client import timeline
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dataname = "davis"
dataname = "kiba"
# 5-fold cross-validation
cross_num = 5
LEARNING_RATE_BASE = 0.0001
# REGULARIZATION_RATE = 0.00001
EPOCH = 3
#
if dataname == "kiba":
    batch_size = 100
    TESTNUM = (118256/5)*4/100

dataname == "davis"
batch_size = 64
TESTNUM = (30056 / 5) * 4 / 100
    
MAX_SEQ_LEN = 1200
MAX_SMI_LEN = 100

Train_path = "./tfrecord/" + dataname + "/train%d.tfrecord"
MODEL_SAVE_PATH = "./" + dataname + "/model%d/"
MODEL_NAME = "model.ckpt"


def parser(record):
    read_features = {
        'drug': tf.compat.v1.FixedLenFeature([MAX_SMI_LEN], dtype=tf.compat.v1.int64),
        'protein': tf.compat.v1.FixedLenFeature([MAX_SEQ_LEN], dtype=tf.compat.v1.int64),
        'affinity': tf.compat.v1.FixedLenFeature([1], dtype=tf.compat.v1.float32)
    }

    read_data = tf.compat.v1.parse_single_example(
        serialized=record, features=read_features)

    drug = tf.compat.v1.cast(read_data['drug'], tf.compat.v1.int32)
    protein = tf.compat.v1.cast(read_data['protein'], tf.compat.v1.int32)
    affinit_y = read_data['affinity']

    return drug, protein, affinit_y


def train(num, train_path):
    with tf.compat.v1.variable_scope("input"):
        dataset = tf.compat.v1.data.TFRecordDataset(train_path)
        dataset = dataset.map(parser)
        dataset = dataset.repeat(EPOCH).shuffle(500).batch(
            batch_size=batch_size)
        train_iterator = dataset.make_initializable_iterator()
        train_drug, train_proteins_to_embeding, train_labels_batch\
            = train_iterator.get_next()

    # regularizer = tf.compat.v1.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    _, _, train_label = \
        model.inference(
            train_drug,
            train_proteins_to_embeding,
            regularizer=None, keep_prob=0.9, trainlabel=1
        )

    global_step = tf.compat.v1.Variable(0, trainable=False)
    with tf.compat.v1.name_scope("train_loss_function"):
        mean_squared_eror = tf.compat.v1.losses.mean_squared_error(
            train_label, train_labels_batch)
        tf.compat.v1.summary.scalar("mean_squared_eror", mean_squared_eror)
        # loss = mean_squared_eror + tf.compat.v1.add_n(tf.compat.v1.get_collection("losses"))
        # tf.compat.v1.summary.scalar("loss", loss)

    with tf.compat.v1.name_scope("train_step"):
        learning_rate = LEARNING_RATE_BASE
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.compat.v1.control_dependencies(update_ops):
            # train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
                mean_squared_eror, global_step=global_step)
            with tf.compat.v1.control_dependencies([train_step]):
                train_op = tf.compat.v1.no_op(name='train')

    merged = tf.compat.v1.summary.merge_all()
    summary_write = tf.compat.v1.summary.FileWriter(
        "./" + dataname + "/path/to/log%d" %
        num, tf.compat.v1.get_default_graph())
    var_list = [var for var in tf.compat.v1.global_variables() if "moving" in var.name]
    var_list += tf.compat.v1.trainable_variables()
    saver = tf.compat.v1.train.Saver(var_list=var_list, max_to_keep=20)

    config = tf.compat.v1.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess,\
            open("./" + dataname + "/path/to/log%d/log.txt" % num, "w") as f:
        print("beginning training")
        sess.run(
            tf.compat.v1.group(
                tf.compat.v1.global_variables_initializer(),
                tf.compat.v1.local_variables_initializer()))
        sess.run(train_iterator.initializer)
        step = 0
        maxloss = 100
        trainMSElist = []
        try:
            while True:
                step += 1
                run_options = tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                #train and test
                # trainLosslist = []
                # Loss, summary, _, MSE, now_step = sess.run(
                #     [loss, merged, train_op, mean_squared_eror, global_step],
                #     options=run_options, run_metadata=run_metadata)
                summary, _, MSE, now_step = sess.run(
                    [merged, train_op, mean_squared_eror, global_step],
                    options=run_options, run_metadata=run_metadata)
                str = "%s-model:%d-step:%d;train_MSE:%g;" % (
                    dataname, num, now_step, MSE)
                f.write(str + "\n")
                trainMSElist.append(MSE)
                # trainLosslist.append(Loss)
                if step % 10 == 0:
                # if step % TESTNUM == 0:
                    summary_write.add_summary(summary, now_step)
                    summary_write.add_run_metadata(
                        run_metadata, tag=(
                            "step%d" %
                            step), global_step=step)
                    trainMSE = 0
                    # trainLoss = 0
                    for i in range(len(trainMSElist)):
                        # trainLoss += trainLosslist[i]
                        trainMSE += trainMSElist[i]
                    # trainLoss /= len(trainLosslist)
                    trainMSE /= len(trainMSElist)
                    # print(
                    #     "%s-model:%d-step:%d;train_Loss:%g;train_MSE:%g." %
                    #     (dataname, num, now_step, trainLoss, trainMSE))
                    print(
                        "%s-model:%d-epoch:%d;step:%d;train_MSE:%g;" %
                        (dataname, num, int(now_step / TESTNUM), now_step, trainMSE))
                    trainMSElist = []
                    if trainMSE < maxloss:
                        saver.save(
                            sess,
                            os.path.join(
                                MODEL_SAVE_PATH %
                                num,
                                MODEL_NAME),
                            global_step=global_step)
                        maxloss = trainMSE
                        print("save model")
                else:
                    pass
        except tf.compat.v1.errors.OutOfRangeError:
            pass
        summary_write.close()


def main(argv=None):
    for i in range(cross_num):
        tf.compat.v1.reset_default_graph()
        if os.path.exists(MODEL_SAVE_PATH % i) is False:
            os.makedirs(MODEL_SAVE_PATH % i)
        print("The No.%d model" % i)
        train_path = Train_path % i

        train(i, train_path)


if __name__ == '__main__':
    tf.compat.v1.app.run()
