# Analyzes the sub folders in the image directory, splits them into stable
# training, testing, and validation sets, and returns a data structure
# describing the lists of images for each label and their paths.
# def main():
#     result = create_image_lists("tf_files/bird_photos/", .2, .4)
#     print("results: ")
#     print(result)
#
# main()
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import tensorflow as tf
import os as os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  positives = 0
  false_positives = 0
  negatives = 0
  false_negatives = 0
  total_test_images = 0
  unable_to_classify = 0
  crow_raven_labels = ["029", "108", "107", "030"]
  image_test_directory = "tf_files/bird_photos_train/"

  # @author Maya Man
  for subdir in os.listdir(image_test_directory):
    if (subdir != ".DS_Store"):
        print("subdir: " +  subdir)
        for f in os.listdir(image_test_directory + subdir):
          if (f != ".DS_Store"):
              print("filename: " + f)
              graph = load_graph(model_file)
              file_name = image_test_directory + subdir + "/" + f
              true_label_number = subdir.split(".")[0]
              t = read_tensor_from_image_file(file_name,
                                              input_height=input_height,
                                              input_width=input_width,
                                              input_mean=input_mean,
                                              input_std=input_std)

              input_name = "import/" + input_layer
              output_name = "import/" + output_layer
              input_operation = graph.get_operation_by_name(input_name);
              output_operation = graph.get_operation_by_name(output_name);
              tic = time.clock()
              with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                  {input_operation.outputs[0]: t})
              results = np.squeeze(results)

              top_k = results.argsort()[-5:][::-1]
              labels = load_labels(label_file)
              toc = time.clock()
            #   print()
              guessed_label_number = labels[top_k[0]].split(" ")[0]
              top_result_confidence = results[top_k[0]]
              print(labels[top_k[0]], results[top_k[0]])
              if (top_result_confidence > 0.5):
                  if ((true_label_number in crow_raven_labels) and (guessed_label_number in crow_raven_labels)):
                    positives = positives + 1
                  elif (not (true_label_number in crow_raven_labels) and (guessed_label_number in crow_raven_labels)):
                    # print("false positives")
                    false_positives = false_positives + 1
                  elif ((true_label_number in crow_raven_labels) and not (guessed_label_number in crow_raven_labels)):
                    # print("false negatives")
                    false_negatives = false_negatives + 1
                  else:
                    # print("negative")
                    negatives = negatives + 1
                  total_test_images = total_test_images + 1
              else:
                unable_to_classify = unable_to_classify + 1
        print("total time", toc - tic)

  success_rate = positives / total_test_images
  failure_rate = (total_test_images - positives) / total_test_images
  print(positives, " positives")
  print(false_positives, " false_positives")
  print(negatives, " negatives")
  print(false_negatives, " false_negatives")
  print(unable_to_classify, "unable_to_classify")
  print(success_rate, " success rate")
  print(failure_rate, " failure rate")
