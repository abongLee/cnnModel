from cnn_model import *
from csv_reader import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('epochs', 100, 'total epoch to train')
tf.app.flags.DEFINE_integer('batchsize', 128, 'batch_size to train')
tf.app.flags.DEFINE_integer('num_classes', 19, 'total classes')
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
tf.app.flags.DEFINE_boolean('is_transfer', False, 'is transfer learning or not')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'checkpoint_model')
tf.app.flags.DEFINE_string('save_new_checkpoint_dir', None, 'checkpoint_model')


def main(argv=None):
    model = cnn_model(epochs=FLAGS.epochs, batchsize=FLAGS.batchsize, num_classes=FLAGS.num_classes,
                      learning_rate=FLAGS.learning_rate, is_transfer=FLAGS.is_transfer,
                      checkpoint_dir=FLAGS.checkpoint_dir, save_new_checkpoint_dir=FLAGS.save_new_checkpoint_dir)
    test_data, test_labels = get_hdf5_testing_data()
    model.eval(images=test_data, labels=test_labels)


if __name__ == '__main__':
    tf.app.run()
