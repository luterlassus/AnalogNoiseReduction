import numpy as np
import tensorflow as tf

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(X, Y, out_path):
    # Args:
    # X             4D numpy array (frames, height, width, chan)
    # Y             4D numpy array (frames, height, width, chan)
    # out_path      File-path for the TFRecords output file.

    print("Converting: " + out_path)

    # Number of images. Used when printing the progress.
    num_images = len(X)

    # Open a TFRecordWriter for the output-file.
    with tf.io.TFRecordWriter(out_path) as writer:

        # Iterate over all the frames
        for i in range(len(X)):
            # Convert the image to raw bytes.
            x_bytes = X[i].tobytes()
            y_bytes = Y[i].tobytes()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'X': wrap_bytes(x_bytes),
                    'Y': wrap_bytes(y_bytes)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
