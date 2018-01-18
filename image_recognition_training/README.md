# Overview

This repo contains code for the
["TensorFlow for poets 2" codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2).

This repo contains a simplified and trimmed down version of tensorflow's
[android image classification example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android)
in the `android/` directory.

The `scripts` directory contains helpers for the codelab.

# Train a model

To retrain a model using a [MobileNet](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) on your own image dataset, you can reference [this tutorial](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#3).

What you need to know is also outlined below.

Pass settings inside Linux shell variables.

`IMAGE_SIZE=224
ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"`

Run the training, replacing `TRAINING_IMAGES_DIRECTORY` with your desired directory inside the `tf_files` folder.

`python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/TRAINING_IMAGES_DIRECTORY`

# Label an image

Run the label script, replacing `IMAGE_FILE_PATH` with the path to the image you wish to label.

This script only labels a single image using the most recent model you trained.

`python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=IMAGE_FILE_PATH`


# Run testing script

The testing script `test_results.py` was written very specifically for tests I was running on the bird data set.

As it is currently written, it attempts to test if every image in the directory specified for the variable `image_test_directory` is a type of crow or raven (defined by the labels stored in the variable `crow_raven_labels`). It calculates the percentage of successful and unsuccessful labelings. It counts the number of positive, false positives, negatives, and false negatives.

To run the script:

`python -m scripts.test_results`

If you have any questions, don't hesitate to email me at [mayaman26@gmail.com](mailto:mayaman26@gmail.com).
