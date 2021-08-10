import tensorflow as tf
import sys

if len(sys.argv) == 1:
    print("Usage: python printModelInfo.py <model(s)>")
else:
    for model in sys.argv[1:]:
        print("Printing stats for model", model)
        m = tf.keras.models.load_model('../Models/' + model + '.h5')
        print(m.summary())

