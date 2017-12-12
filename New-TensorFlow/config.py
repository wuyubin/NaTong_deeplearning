import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", 10, "batch size for training")
tf.flags.DEFINE_integer("epoches", 100, "number of epoches")
tf.flags.DEFINE_integer("disp", 50, "how many iterations to display")
tf.flags.DEFINE_float("weight_decay", 0.001, "weight decay")
tf.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.flags.DEFINE_string("data_path", "F:/natong/natong_deeplearn_data/", "data path storing npy files")
tf.flags.DEFINE_string("data_type", "natong", "data type for training")
tf.flags.DEFINE_string("net_type", "Inception", "data type for training")
tf.flags.DEFINE_string("log_path", "./log/", "log path storing checkpoints")
tf.flags.DEFINE_float("dropout", 0.7, "keep probability of dropout")
tf.flags.DEFINE_string("mode", "train", "train or test")

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
