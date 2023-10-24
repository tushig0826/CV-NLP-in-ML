CV-NLP-in-ML  Image_Captioning_Flickr8K_Dataset.ipynb
Copyright 2018 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
Image captioning with visual attention
 View on TensorFlow.org	 Run in Google Colab	 View source on GitHub	Download notebook
Given an image like the example below, your goal is to generate a caption such as "a surfer riding on a wave".


A man surfing, from wikimedia
The model architecture used here is inspired by Show, Attend and Tell: Neural Image Caption Generation with Visual Attention, but has been updated to use a 2-layer Transformer-decoder. To get the most out of this tutorial you should have some experience with text generation, seq2seq models & attention, or transformers.

The model architecture built in this tutorial is shown below. Features are extracted from the image, and passed to the cross-attention layers of the Transformer-decoder.

The model architecture

The transformer decoder is mainly built from attention layers. It uses self-attention to process the sequence being generated, and it uses cross-attention to attend to the image.

By inspecting the attention weights of the cross attention layers you will see what parts of the image the model is looking at as it generates words.

Prediction

This notebook is an end-to-end example. When you run the notebook, it downloads a dataset, extracts and caches the image features, and trains a decoder model. It then uses the model to generate captions on new images.

Setup
!apt install --allow-change-held-packages libcudnn8=8.1.0.77-1+cuda11.2
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following package was automatically installed and is no longer required:
  libnvidia-common-460
Use 'apt autoremove' to remove it.
The following packages will be REMOVED:
  libcudnn8-dev
The following held packages will be changed:
  libcudnn8
The following packages will be DOWNGRADED:
  libcudnn8
0 upgraded, 0 newly installed, 1 downgraded, 1 to remove and 25 not upgraded.
Need to get 430 MB of archives.
After this operation, 1,392 MB disk space will be freed.
Get:1 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  libcudnn8 8.1.0.77-1+cuda11.2 [430 MB]
Fetched 430 MB in 11s (40.9 MB/s)
(Reading database ... 123942 files and directories currently installed.)
Removing libcudnn8-dev (8.1.1.33-1+cuda11.2) ...
update-alternatives: removing manually selected alternative - switching libcudnn to auto mode
dpkg: warning: downgrading libcudnn8 from 8.1.1.33-1+cuda11.2 to 8.1.0.77-1+cuda11.2
(Reading database ... 123919 files and directories currently installed.)
Preparing to unpack .../libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb ...
Unpacking libcudnn8 (8.1.0.77-1+cuda11.2) over (8.1.1.33-1+cuda11.2) ...
Setting up libcudnn8 (8.1.0.77-1+cuda11.2) ...
!pip uninstall -y tensorflow estimator keras
Found existing installation: tensorflow 2.9.2
Uninstalling tensorflow-2.9.2:
  Successfully uninstalled tensorflow-2.9.2
WARNING: Skipping estimator as it is not installed.
Found existing installation: keras 2.9.0
Uninstalling keras-2.9.0:
  Successfully uninstalled keras-2.9.0
!pip install -U tensorflow_text tensorflow tensorflow_datasets
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tensorflow_text
  Downloading tensorflow_text-2.10.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.9 MB)
     |████████████████████████████████| 5.9 MB 5.1 MB/s 
Collecting tensorflow
  Downloading tensorflow-2.10.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (578.0 MB)
     |████████████████████████████████| 578.0 MB 16 kB/s 
Requirement already satisfied: tensorflow_datasets in /usr/local/lib/python3.7/dist-packages (4.6.0)
Collecting tensorflow_datasets
  Downloading tensorflow_datasets-4.7.0-py3-none-any.whl (4.7 MB)
     |████████████████████████████████| 4.7 MB 49.1 MB/s 
Requirement already satisfied: tensorflow-hub>=0.8.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow_text) (0.12.0)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.1.0)
Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.1.2)
Collecting tensorboard<2.11,>=2.10
  Downloading tensorboard-2.10.1-py3-none-any.whl (5.9 MB)
     |████████████████████████████████| 5.9 MB 54.2 MB/s 
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (14.0.6)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.27.0)
Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.21.6)
Collecting tensorflow-estimator<2.11,>=2.10.0
  Downloading tensorflow_estimator-2.10.0-py2.py3-none-any.whl (438 kB)
     |████████████████████████████████| 438 kB 71.0 MB/s 
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (4.1.1)
Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (0.4.0)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.14.1)
Requirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.17.3)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.50.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (2.0.1)
Collecting keras<2.11,>=2.10.0
  Downloading keras-2.10.0-py2.py3-none-any.whl (1.7 MB)
     |████████████████████████████████| 1.7 MB 57.8 MB/s 
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.15.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from tensorflow) (57.4.0)
Collecting flatbuffers>=2.0
  Downloading flatbuffers-22.10.26-py2.py3-none-any.whl (26 kB)
Requirement already satisfied: packaging in /usr/local/lib/python3.7/dist-packages (from tensorflow) (21.3)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow) (1.3.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow) (1.5.2)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.23.0)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (1.0.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (3.4.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (1.8.1)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (1.35.0)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.6.1)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.4.6)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (4.9)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (4.2.4)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (4.13.0)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (3.9.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.4.8)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2022.9.24)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (1.24.3)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (3.2.2)
Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (1.10.0)
Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (4.64.1)
Requirement already satisfied: importlib-resources in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (5.10.0)
Requirement already satisfied: etils[epath] in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (0.8.0)
Requirement already satisfied: toml in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (0.10.2)
Requirement already satisfied: dill in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (0.3.5.1)
Requirement already satisfied: promise in /usr/local/lib/python3.7/dist-packages (from tensorflow_datasets) (2.3)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging->tensorflow) (3.0.9)
Requirement already satisfied: googleapis-common-protos<2,>=1.52.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-metadata->tensorflow_datasets) (1.56.4)
Installing collected packages: tensorflow-estimator, tensorboard, keras, flatbuffers, tensorflow, tensorflow-text, tensorflow-datasets
  Attempting uninstall: tensorflow-estimator
    Found existing installation: tensorflow-estimator 2.9.0
    Uninstalling tensorflow-estimator-2.9.0:
      Successfully uninstalled tensorflow-estimator-2.9.0
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.9.1
    Uninstalling tensorboard-2.9.1:
      Successfully uninstalled tensorboard-2.9.1
  Attempting uninstall: flatbuffers
    Found existing installation: flatbuffers 1.12
    Uninstalling flatbuffers-1.12:
      Successfully uninstalled flatbuffers-1.12
  Attempting uninstall: tensorflow-datasets
    Found existing installation: tensorflow-datasets 4.6.0
    Uninstalling tensorflow-datasets-4.6.0:
      Successfully uninstalled tensorflow-datasets-4.6.0
Successfully installed flatbuffers-22.10.26 keras-2.10.0 tensorboard-2.10.1 tensorflow-2.10.0 tensorflow-datasets-4.7.0 tensorflow-estimator-2.10.0 tensorflow-text-2.10.0
!pip install einops
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting einops
  Downloading einops-0.5.0-py3-none-any.whl (36 kB)
Installing collected packages: einops
Successfully installed einops-0.5.0
This tutorial uses lots of imports, mostly for loading the dataset(s).

#@title
import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint
Mounting google drive

# mount drive
from google.colab import drive
drive.mount('/content/gdrive')
dir = '/content/gdrive/My Drive/Data'
Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
[Optional] Data handling
This section downloads a captions dataset and prepares it for training. It tokenizes the input text, and caches the results of running all the images through a pretrained feature-extractor model. It's not critical to understand everything in this section.

Toggle section
from google.colab import drive
drive.mount('/content/drive')
---------------------------------------------------------------------------
MessageError                              Traceback (most recent call last)
<ipython-input-1-d5df0069828e> in <module>
      1 from google.colab import drive
----> 2 drive.mount('/content/drive')

/usr/local/lib/python3.7/dist-packages/google/colab/drive.py in mount(mountpoint, force_remount, timeout_ms, readonly)
    104       timeout_ms=timeout_ms,
    105       ephemeral=True,
--> 106       readonly=readonly)
    107 
    108 

/usr/local/lib/python3.7/dist-packages/google/colab/drive.py in _mount(mountpoint, force_remount, timeout_ms, ephemeral, readonly)
    123   if ephemeral:
    124     _message.blocking_request(
--> 125         'request_auth', request={'authType': 'dfs_ephemeral'}, timeout_sec=None)
    126 
    127   mountpoint = _os.path.expanduser(mountpoint)

/usr/local/lib/python3.7/dist-packages/google/colab/_message.py in blocking_request(request_type, request, timeout_sec, parent)
    169   request_id = send_request(
    170       request_type, request, parent=parent, expect_reply=True)
--> 171   return read_reply_from_input(request_id, timeout_sec)

/usr/local/lib/python3.7/dist-packages/google/colab/_message.py in read_reply_from_input(message_id, timeout_sec)
    100         reply.get('colab_msg_id') == message_id):
    101       if 'error' in reply:
--> 102         raise MessageError(reply['error'])
    103       return reply.get('data', None)
    104 

MessageError: Error: credential propagation was unsuccessful
Choose a dataset
This tutorial is set up to give a choice of datasets. Either Flickr8k or a small slice of the Conceptual Captions dataset. These two are downloaded and converted from scratch, but it wouldn't be hard to convert the tutorial to use the caption datasets available in TensorFlow Datasets: Coco Captions and the full Conceptual Captions.

Flickr8k
def flickr8k(path='flickr8k'):
  path = pathlib.Path(path)

  if len(list(path.rglob('*'))) < 16197:
    tf.keras.utils.get_file(
        origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip',
        cache_dir='.',
        cache_subdir=path,
        extract=True)
    tf.keras.utils.get_file(
        origin='https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip',
        cache_dir='.',
        cache_subdir=path,
        extract=True)
    
  captions = (path/"Flickr8k.token.txt").read_text().splitlines()
  captions = (line.split('\t') for line in captions)
  captions = ((fname.split('#')[0], caption) for (fname, caption) in captions)

  cap_dict = collections.defaultdict(list)
  for fname, cap in captions:
    cap_dict[fname].append(cap)

  train_files = (path/'Flickr_8k.trainImages.txt').read_text().splitlines()
  train_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in train_files]

  test_files = (path/'Flickr_8k.testImages.txt').read_text().splitlines()
  test_captions = [(str(path/'Flicker8k_Dataset'/fname), cap_dict[fname]) for fname in test_files]

  train_ds = tf.data.experimental.from_list(train_captions)
  test_ds = tf.data.experimental.from_list(test_captions)

  return train_ds, test_ds
Conceptual Captions
def conceptual_captions(*, data_dir="conceptual_captions", num_train, num_val):
  def iter_index(index_path):
    with open(index_path) as f:
      for line in f:
        caption, url = line.strip().split('\t')
        yield caption, url

  def download_image_urls(data_dir, urls):
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=100)
    def save_image(url):
      hash = hashlib.sha1(url.encode())
      # Name the files after the hash of the URL.
      file_path = data_dir/f'{hash.hexdigest()}.jpeg'
      if file_path.exists():
        # Only download each file once.
        return file_path

      try:
        result = requests.get(url, timeout=5)
      except Exception:
        file_path = None
      else:
        file_path.write_bytes(result.content)
      return file_path
    
    result = []
    out_paths = ex.map(save_image, urls)
    for file_path in tqdm.tqdm(out_paths, total=len(urls)):
      result.append(file_path)

    return result

  def ds_from_index_file(index_path, data_dir, count):
    data_dir.mkdir(exist_ok=True)
    index = list(itertools.islice(iter_index(index_path), count))
    captions = [caption for caption, url in index]
    urls = [url for caption, url in index]

    paths = download_image_urls(data_dir, urls)

    new_captions = []
    new_paths = []
    for cap, path in zip(captions, paths):
      if path is None:
        # Download failed, so skip this pair.
        continue
      new_captions.append(cap)
      new_paths.append(path)
    
    new_paths = [str(p) for p in new_paths]

    ds = tf.data.Dataset.from_tensor_slices((new_paths, new_captions))
    ds = ds.map(lambda path,cap: (path, cap[tf.newaxis])) # 1 caption per image
    return ds

  data_dir = pathlib.Path(data_dir)
  train_index_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/gcc-data/Train/GCC-training.tsv',
    cache_subdir=data_dir,
    cache_dir='.')
  
  val_index_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/gcc-data/Validation/GCC-1.1.0-Validation.tsv',
    cache_subdir=data_dir,
    cache_dir='.')
  
  train_raw = ds_from_index_file(train_index_path, data_dir=data_dir/'train', count=num_train)
  test_raw = ds_from_index_file(val_index_path, data_dir=data_dir/'val', count=num_val)

  return train_raw, test_raw
Download the dataset
The Flickr8k is a good choice because it contains 5-captions per image, more data for a smaller download.

choose = 'flickr8k'

if choose == 'flickr8k':
  train_raw, test_raw = flickr8k()
else:
  train_raw, test_raw = conceptual_captions(num_train=10000, num_val=5000)
The loaders for both datasets above return tf.data.Datasets containing (image_path, captions) pairs. The Flickr8k dataset contains 5 captions per image, while Conceptual Captions has 1:

train_raw.element_spec
for ex_path, ex_captions in train_raw.take(1):
  print(ex_path)
  print(ex_captions)
Image feature extractor
You will use an image model (pretrained on imagenet) to extract the features from each image. The model was trained as an image classifier, but setting include_top=False returns the model without the final classification layer, so you can use the last layer of feature-maps:

IMAGE_SHAPE=(224, 224, 3)
mobilenet = tf.keras.applications.MobileNetV3Small(
    input_shape=IMAGE_SHAPE,
    include_top=False,
    include_preprocessing=True)
mobilenet.trainable=False
Here's a function to load an image and resize it for the model:

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SHAPE[:-1])
    return img
The model returns a feature map for each image in the input batch:

test_img_batch = load_image(ex_path)[tf.newaxis, :]

print(test_img_batch.shape)
print(mobilenet(test_img_batch).shape)
Setup the text tokenizer/vectorizer
You will transform the text captions into integer sequences using the TextVectorization layer, with the following steps:

Use adapt to iterate over all captions, split the captions into words, and compute a vocabulary of the top words.
Tokenize all captions by mapping each word to its index in the vocabulary. All output sequences will be padded to length 50.
Create word-to-index and index-to-word mappings to display results.
def standardize(s):
  s = tf.strings.lower(s)
  s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
  s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
  return s
# Use the top 5000 words for a vocabulary.
vocabulary_size = 5000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    ragged=True)
# Learn the vocabulary from the caption data.
tokenizer.adapt(train_raw.map(lambda fp,txt: txt).unbatch().batch(1024))
tokenizer.get_vocabulary()[:10]
t = tokenizer([['a cat in a hat'], ['a robot dog']])
t
# Create mappings for words to indices and indices to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)
w = index_to_word(t)
w.to_list()
tf.strings.reduce_join(w, separator=' ', axis=-1).numpy()
Prepare the datasets
The train_raw and test_raw datasets contain 1:many (image, captions) pairs.

This function will replicate the image so there are 1:1 images to captions:

def match_shapes(images, captions):
  caption_shape = einops.parse_shape(captions, 'b c')
  captions = einops.rearrange(captions, 'b c -> (b c)')
  images = einops.repeat(
      images, 'b ... -> (b c) ...',
      c = caption_shape['c'])
  return images, captions
for ex_paths, ex_captions in train_raw.batch(32).take(1):
  break

print('image paths:', ex_paths.shape)
print('captions:', ex_captions.shape)
print()

ex_paths, ex_captions = match_shapes(images=ex_paths, captions=ex_captions)

print('image_paths:', ex_paths.shape)
print('captions:', ex_captions.shape)
To be compatible with keras training the dataset should contain (inputs, labels) pairs. For text generation the tokens are both an input and the labels, shifted by one step. This function will convert an (images, texts) pair to an ((images, input_tokens), label_tokens) pair:

def prepare_txt(imgs, txts):
  tokens = tokenizer(txts)

  input_tokens = tokens[..., :-1]
  label_tokens = tokens[..., 1:]
  return (imgs, input_tokens), label_tokens
This function adds operations to a dataset. The steps are:

Load the images (and ignore images that fail to load).
Replicate images to match the number of captions.
Shuffle and rebatch the image, caption pairs.
Tokenize the text, shift the tokens and add label_tokens.
Convert the text from a RaggedTensor representation to padded dense Tensor representation.
def prepare_dataset(ds, tokenizer, batch_size=32, shuffle_buffer=1000):
  # Load the images and make batches.
  ds = (ds
        .shuffle(10000)
        .map(lambda path, caption: (load_image(path), caption))
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size))

  def to_tensor(inputs, labels):
    (images, in_tok), out_tok = inputs, labels
    return (images, in_tok.to_tensor()), out_tok.to_tensor()

  return (ds
          .map(match_shapes, tf.data.AUTOTUNE)
          .unbatch()
          .shuffle(shuffle_buffer)
          .batch(batch_size)
          .map(prepare_txt, tf.data.AUTOTUNE)
          .map(to_tensor, tf.data.AUTOTUNE)
          )
You could install the feature extractor in your model and train on the datasets like this:

train_ds = prepare_dataset(train_raw, tokenizer)
train_ds.element_spec
test_ds = prepare_dataset(test_raw, tokenizer)
test_ds.element_spec
[Optional] Cache the image features
Since the image feature extractor is not changing, and this tutorial is not using image augmentation, the image features can be cached. Same for the text tokenization. The time it takes to set up the cache is earned back on each epoch during training and validation. The code below defines two functions save_dataset and load_dataset:

def save_dataset(ds, save_path, image_model, tokenizer, shards=10, batch_size=32):
  # Load the images and make batches.
  ds = (ds
        .map(lambda path, caption: (load_image(path), caption))
        .apply(tf.data.experimental.ignore_errors())
        .batch(batch_size))

  # Run the feature extractor on each batch
  # Don't do this in a .map, because tf.data runs on the CPU. 
  def gen():
    for (images, captions) in tqdm.tqdm(ds): 
      feature_maps = image_model(images)

      feature_maps, captions = match_shapes(feature_maps, captions)
      yield feature_maps, captions

  # Wrap the generator in a new tf.data.Dataset.
  new_ds = tf.data.Dataset.from_generator(
      gen,
      output_signature=(
          tf.TensorSpec(shape=image_model.output_shape),
          tf.TensorSpec(shape=(None,), dtype=tf.string)))

  # Apply the tokenization 
  new_ds = (new_ds
            .map(prepare_txt, tf.data.AUTOTUNE)
            .unbatch()
            .shuffle(1000))

  # Save the dataset into shard files.
  def shard_func(i, item):
    return i % shards
  new_ds.enumerate().save(save_path, shard_func=shard_func)

def load_dataset(save_path, batch_size=32, shuffle=1000, cycle_length=2):
  def custom_reader_func(datasets):
    datasets = datasets.shuffle(1000)
    return datasets.interleave(lambda x: x, cycle_length=cycle_length)
  
  ds = tf.data.Dataset.load(save_path, reader_func=custom_reader_func)

  def drop_index(i, x):
    return x

  ds = (ds
        .map(drop_index, tf.data.AUTOTUNE)
        .shuffle(shuffle)
        .padded_batch(batch_size)
        .prefetch(tf.data.AUTOTUNE))
  return ds
save_dataset(train_raw, 'train_cache', mobilenet, tokenizer)
save_dataset(test_raw, 'test_cache', mobilenet, tokenizer)
Data ready for training
After those preprocessing steps, here are the datasets:

train_ds = load_dataset('train_cache')
test_ds = load_dataset('test_cache')
train_ds.element_spec
((TensorSpec(shape=(None, 7, 7, 576), dtype=tf.float32, name=None),
  TensorSpec(shape=(None, None), dtype=tf.int64, name=None)),
 TensorSpec(shape=(None, None), dtype=tf.int64, name=None))
The dataset now returns (input, label) pairs suitable for training with keras. The inputs are (images, input_tokens) pairs. The images have been processed with the feature-extractor model. For each location in the input_tokens the model looks at the text so far and tries to predict the next which is lined up at the same location in the labels.

for (inputs, ex_labels) in train_ds.take(1):
  (ex_img, ex_in_tok) = inputs

print(ex_img.shape)
print(ex_in_tok.shape)
print(ex_labels.shape)
(32, 7, 7, 576)
(32, 22)
(32, 22)
The input tokens and the labels are the same, just shifted by 1 step:

print(ex_in_tok[0].numpy())
print(ex_labels[0].numpy())
[   3    2   27   20    8  586    7    2 1752    0    0    0    0    0
    0    0    0    0    0    0    0    0]
[   2   27   20    8  586    7    2 1752    4    0    0    0    0    0
    0    0    0    0    0    0    0    0]
A Transformer decoder model
This model assumes that the pretrained image encoder is sufficient, and just focuses on building the text decoder. This tutorial uses a 2-layer Transformer-decoder.

The implementations are almost identical to those in the Transformers tutorial. Refer back to it for more details.

The Transformer encoder and decoder.

The model will be implemented in three main parts:

Input - The token embedding and positional encoding (SeqEmbedding).
Decoder - A stack of transformer decoder layers (DecoderLayer) where each contains:
A causal self attention later (CausalSelfAttention), where each output location can attend to the output so far.
A cross attention layer (CrossAttention) where each output location can attend to the input image.
A feed forward network (FeedForward) layer which further processes each output location independently.
Output - A multiclass-classification over the output vocabulary.
Input
The input text has already been split up into tokens and converted to sequences of IDs.

Remember that unlike a CNN or RNN the Transformer's attention layers are invariant to the order of the sequence. Without some positional input, it just sees an unordered set not a sequence. So in addition to a simple vector embedding for each token ID, the embedding layer will also include an embedding for each position in the sequence.

The SeqEmbedding layer defined below:

It looks up the embedding vector for each token.
It looks up an embedding vector for each sequence location.
It adds the two together.
It uses mask_zero=True to initialize the keras-masks for the model.
Note: This implementation learns the position embeddings instead of using fixed embeddings like in the Transformer tutorial. Learning the embeddings is slightly less code, but doesn't generalize to longer sequences.

class SeqEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, max_length, depth):
    super().__init__()
    self.pos_embedding = tf.keras.layers.Embedding(input_dim=max_length, output_dim=depth)

    self.token_embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=depth,
        mask_zero=True)
    
    self.add = tf.keras.layers.Add()

  def call(self, seq):
    seq = self.token_embedding(seq) # (batch, seq, depth)

    x = tf.range(tf.shape(seq)[1])  # (seq)
    x = x[tf.newaxis, :]  # (1, seq)
    x = self.pos_embedding(x)  # (1, seq, depth)

    return self.add([seq,x])
Decoder
The decoder is a standard Transformer-decoder, it contains a stack of DecoderLayers where each contains three sublayers: a CausalSelfAttention, a CrossAttention, and aFeedForward. The implementations are almost identical to the Transformer tutorial, refer to it for more details.

The CausalSelfAttention layer is below:

class CausalSelfAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    # Use Add instead of + so the keras mask propagates through.
    self.add = tf.keras.layers.Add() 
    self.layernorm = tf.keras.layers.LayerNormalization()
  
  def call(self, x):
    attn = self.mha(query=x, value=x,
                    use_causal_mask=True)
    x = self.add([x, attn])
    return self.layernorm(x)
The CrossAttention layer is below. Note the use of return_attention_scores.

class CrossAttention(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.add = tf.keras.layers.Add() 
    self.layernorm = tf.keras.layers.LayerNormalization()
  
  def call(self, x, y, **kwargs):
    attn, attention_scores = self.mha(
             query=x, value=y,
             return_attention_scores=True)
    
    self.last_attention_scores = attention_scores

    x = self.add([x, attn])
    return self.layernorm(x)
The FeedForward layer is below. Remember that a layers.Dense layer is applied to the last axis of the input. The input will have a shape of (batch, sequence, channels), so it automatically applies pointwise across the batch and sequence axes.

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, units, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
        tf.keras.layers.Dense(units=2*units, activation='relu'),
        tf.keras.layers.Dense(units=units),
        tf.keras.layers.Dropout(rate=dropout_rate),
    ])

    self.layernorm = tf.keras.layers.LayerNormalization()
  
  def call(self, x):
    x = x + self.seq(x)
    return self.layernorm(x)
Next arrange these three layers into a larger DecoderLayer. Each decoder layer applies the three smaller layers in sequence. After each sublayer the shape of out_seq is (batch, sequence, channels). The decoder layer also returns the attention_scores for later visualizations.

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, units, num_heads=1, dropout_rate=0.1):
    super().__init__()
    
    self.self_attention = CausalSelfAttention(num_heads=num_heads,
                                              key_dim=units,
                                              dropout=dropout_rate)
    self.cross_attention = CrossAttention(num_heads=num_heads,
                                          key_dim=units,
                                          dropout=dropout_rate)
    self.ff = FeedForward(units=units, dropout_rate=dropout_rate)
      

  def call(self, inputs, training=False):
    in_seq, out_seq = inputs

    # Text input
    out_seq = self.self_attention(out_seq)

    out_seq = self.cross_attention(out_seq, in_seq)
    
    self.last_attention_scores = self.cross_attention.last_attention_scores

    out_seq = self.ff(out_seq)

    return out_seq
Output
At minimum the output layer needs a layers.Dense layer to generate logit-predictions for each token at each location.

But there are a few other features you can add to make this work a little better:

Handle bad tokens: The model will be generating text. It should never generate a pad, unknown, or start token ('', '[UNK]', '[START]'). So set the bias for these to a large negative value.

Note: You'll need to ignore these tokens in the loss function as well.

Smart initialization: The default initialization of a dense layer will

give a model that initially predicts each token with almost uniform likelihood. The actual token distribution is far from uniform. The optimal value for the initial bias of the output layer is the log of the probability of each token. So include an adapt method to count the tokens and set the optimal initial bias. This reduces the initial loss from the entropy of the uniform distribution (log(vocabulary_size)) to the marginal entropy of the distribution (-p*log(p)).

#@title
class TokenOutput(tf.keras.layers.Layer):
  def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
    super().__init__()
    
    self.dense = tf.keras.layers.Dense(
        units=tokenizer.vocabulary_size(), **kwargs)
    self.tokenizer = tokenizer
    self.banned_tokens = banned_tokens

    self.bias = None

  def adapt(self, ds):
    counts = collections.Counter()
    vocab_dict = {name: id 
                  for id, name in enumerate(self.tokenizer.get_vocabulary())}

    for tokens in tqdm.tqdm(ds):
      counts.update(tokens.numpy().flatten())

    counts_arr = np.zeros(shape=(self.tokenizer.vocabulary_size(),))
    counts_arr[np.array(list(counts.keys()), dtype=np.int32)] = list(counts.values())

    counts_arr = counts_arr[:]
    for token in self.banned_tokens:
      counts_arr[vocab_dict[token]] = 0

    total = counts_arr.sum()
    p = counts_arr/total
    p[counts_arr==0] = 1.0
    log_p = np.log(p)  # log(1) == 0

    entropy = -(log_p*p).sum()

    print()
    print(f"Uniform entropy: {np.log(self.tokenizer.vocabulary_size()):0.2f}")
    print(f"Marginal entropy: {entropy:0.2f}")

    self.bias = log_p
    self.bias[counts_arr==0] = -1e9

  def call(self, x):
    x = self.dense(x)
    # TODO(b/250038731): Fix this.
    # An Add layer doesn't work because of the different shapes.
    # This clears the mask, that's okay because it prevents keras from rescaling
    # the losses.
    return x + self.bias
The smart initialization will significantly reduce the initial loss:

output_layer = TokenOutput(tokenizer, banned_tokens=('', '[UNK]', '[START]'))
# This might run a little faster if the dataset didn't also have to load the image data.
output_layer.adapt(train_ds.map(lambda inputs, labels: labels))
100%|██████████| 938/938 [00:10<00:00, 91.57it/s] 
Uniform entropy: 8.52
Marginal entropy: 5.29
Build the model
To build the model, you need to combine several parts:

The image feature_extractor and the text tokenizer and.
The seq_embedding layer, to convert batches of token-IDs to vectors (batch, sequence, channels).
The stack of DecoderLayers layers that will process the text and image data.
The output_layer which returns a pointwise prediction of what the next word should be.
class Captioner(tf.keras.Model):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, tokenizer, feature_extractor, output_layer, num_layers=1,
               units=256, max_length=50, num_heads=1, dropout_rate=0.1):
    super().__init__()
    self.feature_extractor = feature_extractor
    self.tokenizer = tokenizer
    self.word_to_index = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())
    self.index_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True) 

    self.seq_embedding = SeqEmbedding(
        vocab_size=tokenizer.vocabulary_size(),
        depth=units,
        max_length=max_length)

    self.decoder_layers = [
        DecoderLayer(units, num_heads=num_heads, dropout_rate=dropout_rate)
        for n in range(num_layers)]

    self.output_layer = output_layer
When you call the model, for training, it receives an image, txt pair. To make this function more usable, be flexible about the input:

If the image has 3 channels run it through the feature_extractor. Otherwise assume that it has been already. Similarly
If the text has dtype tf.string run it through the tokenizer.
After that running the model is only a few steps:

Flatten the extracted image features, so they can be input to the decoder layers.
Look up the token embeddings.
Run the stack of DecoderLayers, on the image features and text embeddings.
Run the output layer to predict the next token at each position.
  @Captioner.add_method
  def call(self, inputs):
    image, txt = inputs

    if image.shape[-1] == 3:
      # Apply the feature-extractor, if you get an RGB image.
      image = self.feature_extractor(image)
    
    # Flatten the feature map
    image = einops.rearrange(image, 'b h w c -> b (h w) c')


    if txt.dtype == tf.string:
      # Apply the tokenizer if you get string inputs.
      txt = tokenizer(txt)

    txt = self.seq_embedding(txt)

    # Look at the image
    for dec_layer in self.decoder_layers:
      txt = dec_layer(inputs=(image, txt))
      
    txt = self.output_layer(txt)

    return txt
model = Captioner(tokenizer, feature_extractor=mobilenet, output_layer=output_layer,
                  units=256, dropout_rate=0.5, num_layers=2, num_heads=2)
Generate captions
Before getting into training, write a bit of code to generate captions. You'll use this to see how training is progressing.

Start by downloading a test image:

image_url = 'https://tensorflow.org/images/surf.jpg'
image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
image = load_image(image_path)
Downloading data from https://tensorflow.org/images/surf.jpg
64400/64400 [==============================] - 0s 3us/step
To caption an image with this model:

Extract the img_features
Initialize the list of output tokens with a [START] token.
Pass img_features and tokens into the model.
It returns a list of logits.
Choose the next token based on those logits.
Add it to the list of tokens, and continue the loop.
If it generates an '[END]' token, break out of the loop.
So add a "simple" method to do just that:

@Captioner.add_method
def simple_gen(self, image, temperature=1):
  initial = self.word_to_index([['[START]']]) # (batch, sequence)
  img_features = self.feature_extractor(image[tf.newaxis, ...])

  tokens = initial # (batch, sequence)
  for n in range(50):
    preds = self((img_features, tokens)).numpy()  # (batch, sequence, vocab)
    preds = preds[:,-1, :]  #(batch, vocab)
    if temperature==0:
        next = tf.argmax(preds, axis=-1)[:, tf.newaxis]  # (batch, 1)
    else:
        next = tf.random.categorical(preds/temperature, num_samples=1)  # (batch, 1)
    tokens = tf.concat([tokens, next], axis=1) # (batch, sequence) 

    if next[0] == self.word_to_index('[END]'):
      break
  words = index_to_word(tokens[0, 1:-1])
  result = tf.strings.reduce_join(words, axis=-1, separator=' ')
  return result.numpy().decode()
Here are some generated captions for that image, the model's untrained, so they don't make much sense yet:

for t in (0.0, 0.5, 1.0):
  result = model.simple_gen(image, temperature=t)
  print(result)
a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a
a a a a in
and up the
The temperature parameter allows you to interpolate between 3 modes:

Greedy decoding (temperature=0.0) - Chooses the most likely next token at each step.
Random sampling according to the logits (temperature=1.0).
Uniform random sampling (temperature >> 1.0).
Since the model is untrained, and it used the frequency-based initialization, the "greedy" output (first) usually only contains the most common tokens: ['a', '.', '[END]'].

Train
To train the model you'll need several additional components:

The Loss and metrics
The Optimizer
Optional Callbacks
Losses and metrics
Here's an implementation of a masked loss and accuracy:

When calculating the mask for the loss, note the loss < 1e8. This term discards the artificial, impossibly high losses for the banned_tokens.

def masked_loss(labels, preds):  
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, preds)

  mask = (labels != 0) & (loss < 1e8) 
  mask = tf.cast(mask, loss.dtype)

  loss = loss*mask
  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def masked_acc(labels, preds):
  mask = tf.cast(labels!=0, tf.float32)
  preds = tf.argmax(preds, axis=-1)
  labels = tf.cast(labels, tf.int64)
  match = tf.cast(preds == labels, mask.dtype)
  acc = tf.reduce_sum(match*mask)/tf.reduce_sum(mask)
  return acc
Callbacks
For feedback during training setup a keras.callbacks.Callback to generate some captions for the surfer image at the end of each epoch.

class GenerateText(tf.keras.callbacks.Callback):
  def __init__(self):
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_path = tf.keras.utils.get_file('surf.jpg', origin=image_url)
    self.image = load_image(image_path)

  def on_epoch_end(self, epochs=None, logs=None):
    print()
    print()
    for t in (0.0, 0.5, 1.0):
      result = self.model.simple_gen(self.image, temperature=t)
      print(result)
    print()
It generates three output strings, like the earlier example, like before the first is "greedy", choosing the argmax of the logits at each step.

g = GenerateText()
g.model = model
g.on_epoch_end(0)

a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a a

young down swords a a of in a a mask wall in soccer yellow

Saving generated text model

# This is formatted as code
g.model.save(f"{dir}/model")
WARNING:tensorflow:Skipping full serialization of Keras layer <keras.layers.preprocessing.string_lookup.StringLookup object at 0x7f8c0a05afd0>, because it is not built.
WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 5 of 148). These functions will not be directly callable after loading.
Also use callbacks.EarlyStopping to terminate training when the model starts to overfit.

callbacks = [
    GenerateText(),
    tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True)]
Train
Configure and execute the training.

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
           loss=masked_loss,
           metrics=[masked_acc])
For more frequent reporting, use the Dataset.repeat() method, and set the steps_per_epoch and validation_steps arguments to Model.fit.

With this setup on Flickr8k a full pass over the dataset is 900+ batches, but below the reporting-epochs are 100 steps.

history = model.fit(
    train_ds.repeat(),
    steps_per_epoch=100,
    validation_data=test_ds.repeat(),
    validation_steps=20,
    epochs=100,
    callbacks=callbacks)
Epoch 1/100
100/100 [==============================] - ETA: 0s - loss: 5.0142 - masked_acc: 0.1961

a man in a man in a man in a
a man is in a dog
the man young is motorcycle hands rainbow

100/100 [==============================] - 11s 73ms/step - loss: 5.0142 - masked_acc: 0.1961 - val_loss: 4.6410 - val_masked_acc: 0.2431
Epoch 2/100
 99/100 [============================>.] - ETA: 0s - loss: 4.6250 - masked_acc: 0.2514

a man in a red is is is is is is
a man is is on a woman
a field wearing a

100/100 [==============================] - 6s 61ms/step - loss: 4.6224 - masked_acc: 0.2514 - val_loss: 4.4354 - val_masked_acc: 0.2674
Epoch 3/100
100/100 [==============================] - ETA: 0s - loss: 4.4085 - masked_acc: 0.2761

a man in a red is is running in the water
a man in a yellow is running on the water
a streets with another girl

100/100 [==============================] - 6s 65ms/step - loss: 4.4085 - masked_acc: 0.2761 - val_loss: 4.1824 - val_masked_acc: 0.2897
Epoch 4/100
100/100 [==============================] - ETA: 0s - loss: 4.2429 - masked_acc: 0.2940

a man in a red shirt is running in the water
a man is wearing a red is running in the beach
a little boy wooded two faces instrument people retrieves

100/100 [==============================] - 7s 68ms/step - loss: 4.2429 - masked_acc: 0.2940 - val_loss: 4.0760 - val_masked_acc: 0.3035
Epoch 5/100
 99/100 [============================>.] - ETA: 0s - loss: 4.1342 - masked_acc: 0.3050

a man in a red shirt is running in the water
a person in two people in the water
a child is with a small jacket holding a surfboard

100/100 [==============================] - 7s 67ms/step - loss: 4.1326 - masked_acc: 0.3050 - val_loss: 3.9179 - val_masked_acc: 0.3226
Epoch 6/100
 99/100 [============================>.] - ETA: 0s - loss: 4.0327 - masked_acc: 0.3143

a man in a red shirt is running in the water
a man is wearing a large yellow shirt is in the water
a dirt show in an hangs a tent

100/100 [==============================] - 7s 66ms/step - loss: 4.0316 - masked_acc: 0.3144 - val_loss: 3.7998 - val_masked_acc: 0.3238
Epoch 7/100
 99/100 [============================>.] - ETA: 0s - loss: 3.9180 - masked_acc: 0.3269

a man in a red shirt is running in the water
a man in a red shirt is running through a water
a little this boy in the up

100/100 [==============================] - 6s 64ms/step - loss: 3.9190 - masked_acc: 0.3270 - val_loss: 3.8015 - val_masked_acc: 0.3310
Epoch 8/100
 99/100 [============================>.] - ETA: 0s - loss: 3.9015 - masked_acc: 0.3280

a man in a red shirt is jumping in the water
a man a person in a pool
a rock downhill a setting on the background

100/100 [==============================] - 6s 63ms/step - loss: 3.8990 - masked_acc: 0.3282 - val_loss: 3.7025 - val_masked_acc: 0.3352
Epoch 9/100
 99/100 [============================>.] - ETA: 0s - loss: 3.8079 - masked_acc: 0.3357

a man in a red shirt is running in the water
a man in a red shirt is wearing a red pool
two girls on a field with water

100/100 [==============================] - 7s 67ms/step - loss: 3.8064 - masked_acc: 0.3360 - val_loss: 3.7057 - val_masked_acc: 0.3307
Epoch 10/100
 99/100 [============================>.] - ETA: 0s - loss: 3.7148 - masked_acc: 0.3402

a man in a red shirt is jumping in a pool
a person is wearing a blue shirt is jumping in the pool
two kids in a body of a stick over jean object in agility water

100/100 [==============================] - 7s 69ms/step - loss: 3.7154 - masked_acc: 0.3403 - val_loss: 3.6071 - val_masked_acc: 0.3350
Epoch 11/100
 99/100 [============================>.] - ETA: 0s - loss: 3.6580 - masked_acc: 0.3461

a man in a red shirt is jumping into a pool
a man in a red shirt is running through a pool
a man is wearing a shallow blue has in the lips posing in the flies are waiting

100/100 [==============================] - 7s 73ms/step - loss: 3.6592 - masked_acc: 0.3460 - val_loss: 3.5146 - val_masked_acc: 0.3497
Epoch 12/100
 99/100 [============================>.] - ETA: 0s - loss: 3.6060 - masked_acc: 0.3489

a man is jumping into the water
a man in a red shirt is running through a pool
a person doing a bikers is running in sunglasses at a blue

100/100 [==============================] - 7s 66ms/step - loss: 3.6067 - masked_acc: 0.3489 - val_loss: 3.5154 - val_masked_acc: 0.3478
Epoch 13/100
100/100 [==============================] - ETA: 0s - loss: 3.5836 - masked_acc: 0.3511

a man in a blue shirt is jumping over a pool
a little boy is jumping over a wave
a large sidewalk in her is being tennis

100/100 [==============================] - 8s 82ms/step - loss: 3.5836 - masked_acc: 0.3511 - val_loss: 3.4462 - val_masked_acc: 0.3525
Epoch 14/100
 99/100 [============================>.] - ETA: 0s - loss: 3.5470 - masked_acc: 0.3532

a man is jumping into the water
a boy in a red shirt is jumping on a blue wave
a tan standing in red is standing and a wave

100/100 [==============================] - 7s 71ms/step - loss: 3.5486 - masked_acc: 0.3527 - val_loss: 3.4658 - val_masked_acc: 0.3514
Epoch 15/100
 99/100 [============================>.] - ETA: 0s - loss: 3.5029 - masked_acc: 0.3551

a man in a blue shirt is playing in the water
a person wearing a red shirt is standing in a wave
a person stands in a swimming by swimming pool

100/100 [==============================] - 7s 67ms/step - loss: 3.5012 - masked_acc: 0.3552 - val_loss: 3.3968 - val_masked_acc: 0.3607
Epoch 16/100
 99/100 [============================>.] - ETA: 0s - loss: 3.4512 - masked_acc: 0.3601

a man is jumping into the water
a man is wearing a white shirt is jumping into a snowy pool
the surfer is airborne atv in the ocean pool as a blue striped him

100/100 [==============================] - 7s 68ms/step - loss: 3.4480 - masked_acc: 0.3605 - val_loss: 3.3830 - val_masked_acc: 0.3632
Epoch 17/100
 99/100 [============================>.] - ETA: 0s - loss: 3.4406 - masked_acc: 0.3648

a man in a red shirt is jumping into a pool
a man in a red jacket is running through a water
a baby in a red through the water

100/100 [==============================] - 6s 64ms/step - loss: 3.4402 - masked_acc: 0.3646 - val_loss: 3.4307 - val_masked_acc: 0.3488
Epoch 18/100
100/100 [==============================] - ETA: 0s - loss: 3.4322 - masked_acc: 0.3623

a man is in a wave
a man is jumping into the water
a woman is climbs a face by the surfer

100/100 [==============================] - 6s 61ms/step - loss: 3.4322 - masked_acc: 0.3623 - val_loss: 3.2798 - val_masked_acc: 0.3650
Epoch 19/100
 99/100 [============================>.] - ETA: 0s - loss: 3.3970 - masked_acc: 0.3633

a man in a red shirt is jumping into a pool
a man in a blue water
a older man stands on a water knit neon swimming pool

100/100 [==============================] - 6s 63ms/step - loss: 3.3927 - masked_acc: 0.3637 - val_loss: 3.3663 - val_masked_acc: 0.3529
Epoch 20/100
 99/100 [============================>.] - ETA: 0s - loss: 3.2909 - masked_acc: 0.3723

a man in a red shirt is jumping into the water
a man is climbing a wave
a man playing in a rocky wave

100/100 [==============================] - 6s 61ms/step - loss: 3.2902 - masked_acc: 0.3723 - val_loss: 3.2500 - val_masked_acc: 0.3728
Epoch 21/100
 99/100 [============================>.] - ETA: 0s - loss: 3.2801 - masked_acc: 0.3759

a man in a red shirt is swimming pool
a girl in a blue shirt is sitting on the water
a girl running through the ocean

100/100 [==============================] - 6s 62ms/step - loss: 3.2819 - masked_acc: 0.3759 - val_loss: 3.2985 - val_masked_acc: 0.3692
Epoch 22/100
 99/100 [============================>.] - ETA: 0s - loss: 3.2666 - masked_acc: 0.3749

a man in a blue shirt is swimming pool
a man in a blue shirt is surfing
four people play outside into a pool next to the pool

100/100 [==============================] - 7s 66ms/step - loss: 3.2671 - masked_acc: 0.3752 - val_loss: 3.1768 - val_masked_acc: 0.3731
Epoch 23/100
100/100 [==============================] - ETA: 0s - loss: 3.1987 - masked_acc: 0.3781

a man in a red shirt is swimming pool
a man is doing a into the water
a shirtless man in a surfing a surfer while tripod

100/100 [==============================] - 7s 69ms/step - loss: 3.1987 - masked_acc: 0.3781 - val_loss: 3.1759 - val_masked_acc: 0.3676
Epoch 24/100
100/100 [==============================] - ETA: 0s - loss: 3.2113 - masked_acc: 0.3801

a man in a blue shirt is swimming in the water
a man in a red jacket is swimming pool
a young child is swimming a wave

100/100 [==============================] - 7s 67ms/step - loss: 3.2113 - masked_acc: 0.3801 - val_loss: 3.2139 - val_masked_acc: 0.3680
Epoch 25/100
 99/100 [============================>.] - ETA: 0s - loss: 3.2069 - masked_acc: 0.3785

a man in a red shirt is swimming pool
a person in a red suit is swimming pool
a man in a orange shorts is doing a blue surfboard

100/100 [==============================] - 6s 64ms/step - loss: 3.2056 - masked_acc: 0.3787 - val_loss: 3.2654 - val_masked_acc: 0.3681
Epoch 26/100
 99/100 [============================>.] - ETA: 0s - loss: 3.1912 - masked_acc: 0.3819

a man in a red shirt is surfing
a man in blue shirt is climbing a wave
a cyclist wearing a helmet runs toward a bridge

100/100 [==============================] - 6s 63ms/step - loss: 3.1903 - masked_acc: 0.3821 - val_loss: 3.1653 - val_masked_acc: 0.3793
Epoch 27/100
 99/100 [============================>.] - ETA: 0s - loss: 3.1653 - masked_acc: 0.3834

a man in a blue shirt is riding a wave
a man is riding a wave
a shirtless man in a red wetsuit jumping through the water

100/100 [==============================] - 6s 63ms/step - loss: 3.1659 - masked_acc: 0.3832 - val_loss: 3.1150 - val_masked_acc: 0.3800
Epoch 28/100
 99/100 [============================>.] - ETA: 0s - loss: 3.1692 - masked_acc: 0.3812

a man in a red shirt is swimming pool
a person is riding a surfboard in a wave
the toddler in the air into a person in the ocean

100/100 [==============================] - 6s 63ms/step - loss: 3.1692 - masked_acc: 0.3813 - val_loss: 3.1195 - val_masked_acc: 0.3793
Epoch 29/100
 99/100 [============================>.] - ETA: 0s - loss: 3.0891 - masked_acc: 0.3875

a man in a blue wetsuit is surfing
a surfer is coming out of a wave
a wave off a flying wave

100/100 [==============================] - 6s 59ms/step - loss: 3.0887 - masked_acc: 0.3873 - val_loss: 3.1233 - val_masked_acc: 0.3790
Epoch 30/100
 99/100 [============================>.] - ETA: 0s - loss: 3.0482 - masked_acc: 0.3912

a man in a red shirt is surfing
a man in a red wave
a surfer riding their motorcycle beam

100/100 [==============================] - 6s 57ms/step - loss: 3.0479 - masked_acc: 0.3914 - val_loss: 3.1299 - val_masked_acc: 0.3824
Epoch 31/100
 99/100 [============================>.] - ETA: 0s - loss: 3.0376 - masked_acc: 0.3936

a man in a red shirt is surfing
a man in red shirt is surfing
a man in a blue helmet riding a a rock

100/100 [==============================] - 6s 62ms/step - loss: 3.0386 - masked_acc: 0.3934 - val_loss: 3.0202 - val_masked_acc: 0.3959
Epoch 32/100
 99/100 [============================>.] - ETA: 0s - loss: 3.0180 - masked_acc: 0.3959

a surfer is surfing on a wave
a surfer in a red shirt plays in a pool
a surfer is riding through the surf

100/100 [==============================] - 6s 60ms/step - loss: 3.0181 - masked_acc: 0.3956 - val_loss: 3.0737 - val_masked_acc: 0.3818
Epoch 33/100
 99/100 [============================>.] - ETA: 0s - loss: 3.0336 - masked_acc: 0.3922

a man in a red shirt is surfing
a surfer in a red kayak in the water
a girl on a teen boat made bottom

100/100 [==============================] - 6s 61ms/step - loss: 3.0357 - masked_acc: 0.3919 - val_loss: 3.0290 - val_masked_acc: 0.3813
Epoch 34/100
100/100 [==============================] - ETA: 0s - loss: 2.9951 - masked_acc: 0.3969

a man in a red shirt is surfing
a man in a red helmet is surfing
a child with a black and white goes through a red bandanna

100/100 [==============================] - 6s 65ms/step - loss: 2.9951 - masked_acc: 0.3969 - val_loss: 3.0584 - val_masked_acc: 0.3846
Epoch 35/100
100/100 [==============================] - ETA: 0s - loss: 2.9764 - masked_acc: 0.3983

a man in a red shirt is surfing
a man is riding a wave
a man holds a boat on the wave

100/100 [==============================] - 7s 66ms/step - loss: 2.9764 - masked_acc: 0.3983 - val_loss: 3.0075 - val_masked_acc: 0.3923
Epoch 36/100
100/100 [==============================] - ETA: 0s - loss: 3.0037 - masked_acc: 0.3966

a man in a red shirt is surfing
a man in a blue shirt is surfing on a wave
a surfer jumping off a wave

100/100 [==============================] - 7s 67ms/step - loss: 3.0037 - masked_acc: 0.3966 - val_loss: 3.0673 - val_masked_acc: 0.3879
Epoch 37/100
100/100 [==============================] - ETA: 0s - loss: 2.9891 - masked_acc: 0.3969

a man in a red shirt is surfing
a man in a red jacket is surfing
a surfer the driver wearing a wave

100/100 [==============================] - 9s 88ms/step - loss: 2.9891 - masked_acc: 0.3969 - val_loss: 2.9470 - val_masked_acc: 0.3979
Epoch 38/100
 99/100 [============================>.] - ETA: 0s - loss: 2.9411 - masked_acc: 0.4019

a man in a yellow shirt is surfing
a man in a yellow kayak
a person in orange bathing suit is catching a wave

100/100 [==============================] - 6s 59ms/step - loss: 2.9406 - masked_acc: 0.4019 - val_loss: 2.9690 - val_masked_acc: 0.3928
Epoch 39/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8678 - masked_acc: 0.4126

a surfer rides a wave
a man in a red shirt and white wave
a girl wearing a white beard holding a big wave

100/100 [==============================] - 6s 61ms/step - loss: 2.8698 - masked_acc: 0.4125 - val_loss: 2.9198 - val_masked_acc: 0.4012
Epoch 40/100
100/100 [==============================] - ETA: 0s - loss: 2.8600 - masked_acc: 0.4098

a man in a red shirt is surfing
a man wearing a red kayak in the water
a person in a red surfs into the canoe

100/100 [==============================] - 6s 62ms/step - loss: 2.8600 - masked_acc: 0.4098 - val_loss: 2.9218 - val_masked_acc: 0.3966
Epoch 41/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8636 - masked_acc: 0.4071

a man in a red shirt is surfing
a man in a yellow kayak in a swimming pool
a man sits in a shallow water

100/100 [==============================] - 6s 60ms/step - loss: 2.8628 - masked_acc: 0.4072 - val_loss: 2.9555 - val_masked_acc: 0.3915
Epoch 42/100
 99/100 [============================>.] - ETA: 0s - loss: 2.9041 - masked_acc: 0.4065

a man in a red shirt is surfing
a surfer in a red kayak is surfing
a little boy in a dark dark orange wave

100/100 [==============================] - 6s 64ms/step - loss: 2.9032 - masked_acc: 0.4066 - val_loss: 2.9012 - val_masked_acc: 0.3957
Epoch 43/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8547 - masked_acc: 0.4106

a man in a red shirt is surfing
a surfer rides a surfboard
man surfing out asian with a red saddle

100/100 [==============================] - 6s 59ms/step - loss: 2.8511 - masked_acc: 0.4108 - val_loss: 3.0047 - val_masked_acc: 0.3822
Epoch 44/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8611 - masked_acc: 0.4094

a man in a red shirt is surfing a wave
a surfer rides a wave
a man in a yellow mask wrapped on the bathing suit is on the litter a clear blue waves

100/100 [==============================] - 7s 67ms/step - loss: 2.8619 - masked_acc: 0.4092 - val_loss: 2.9587 - val_masked_acc: 0.3893
Epoch 45/100
100/100 [==============================] - ETA: 0s - loss: 2.8370 - masked_acc: 0.4127

a surfer rides a wave
a surfer rides a wave
a surfer in her wave high goggles while someone is wakeboarding

100/100 [==============================] - 6s 58ms/step - loss: 2.8370 - masked_acc: 0.4127 - val_loss: 2.9373 - val_masked_acc: 0.3948
Epoch 46/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8282 - masked_acc: 0.4109

a surfer rides a wave
a surfer is in the air with a wave
a man in a red shirt is dribbling the water

100/100 [==============================] - 6s 60ms/step - loss: 2.8267 - masked_acc: 0.4111 - val_loss: 2.9097 - val_masked_acc: 0.3978
Epoch 47/100
 99/100 [============================>.] - ETA: 0s - loss: 2.8380 - masked_acc: 0.4096

a surfer in a red wetsuit is surfing
a surfer in a red wetsuit is surfing in the ocean
a surfer on a wave smiling into the water on the waves

100/100 [==============================] - 7s 68ms/step - loss: 2.8378 - masked_acc: 0.4097 - val_loss: 2.9170 - val_masked_acc: 0.3950
Plot the loss and accuracy over the training run:

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.savefig(f"{dir}/loss.png")

plt.plot(history.history['masked_acc'], label='accuracy')
plt.plot(history.history['val_masked_acc'], label='val_accuracy')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('CE/token')
plt.legend()
plt.savefig(f"{dir}/accuracy.png")

--> Saving the plots

#history variable export to excel
Attention plots
Now, using the trained model, run that simple_gen method on the image:

result = model.simple_gen(image, temperature=0.0)
result
'a man in a red shirt is surfing'
Split the output back into tokens:

str_tokens = result.split()
str_tokens.append('[END]')
The DecoderLayers each cache the attention scores for their CrossAttention layer. The shape of each attention map is (batch=1, heads, sequence, image):

attn_maps = [layer.last_attention_scores for layer in model.decoder_layers]
[map.shape for map in attn_maps]
[TensorShape([1, 2, 9, 49]), TensorShape([1, 2, 9, 49])]
So stack the maps along the batch axis, then average over the (batch, heads) axes, while splitting the image axis back into height, width:

attention_maps = tf.concat(attn_maps, axis=0)
attention_maps = einops.reduce(
    attention_maps,
    'batch heads sequence (height width) -> sequence height width',
    height=7, width=7,
    reduction='mean')
Now you have a single attention map, for each sequence prediction. The values in each map should sum to 1.

einops.reduce(attention_maps, 'sequence height width -> sequence', reduction='sum')
<tf.Tensor: shape=(9,), dtype=float32, numpy=
array([1.        , 1.        , 1.        , 1.        , 1.        ,
       1.        , 0.99999994, 1.        , 1.        ], dtype=float32)>
So here is where the model was focusing attention while generating each token of the output:

def plot_attention_maps(image, str_tokens, attention_map):
    fig = plt.figure(figsize=(16, 9))

    len_result = len(str_tokens)
    
    titles = []
    for i in range(len_result):
      map = attention_map[i]
      grid_size = max(int(np.ceil(len_result/2)), 2)
      ax = fig.add_subplot(3, grid_size, i+1)
      titles.append(ax.set_title(str_tokens[i]))
      img = ax.imshow(image)
      ax.imshow(map, cmap='gray', alpha=0.6, extent=img.get_extent(),
                clim=[0.0, np.max(map)])

    plt.tight_layout()
plot_attention_maps(image/255, str_tokens, attention_maps)

Now put that together into a more usable function:

@Captioner.add_method
def run_and_show_attention(self, image, temperature=0.0):
  result_txt = self.simple_gen(image, temperature)
  str_tokens = result_txt.split()
  str_tokens.append('[END]')

  attention_maps = [layer.last_attention_scores for layer in self.decoder_layers]
  attention_maps = tf.concat(attention_maps, axis=0)
  attention_maps = einops.reduce(
      attention_maps,
      'batch heads sequence (height width) -> sequence height width',
      height=7, width=7,
      reduction='mean')
  
  plot_attention_maps(image/255, str_tokens, attention_maps)
  t = plt.suptitle(result_txt)
  text_file = open(f"{dir}/result.txt", "w")
  text_file.write(result_txt)
  text_file.close()
  #str(result_txt).save(f"{dir}/result.txt")
  t.set_y(1.05)
run_and_show_attention(model, image)

Try it on your own images
For fun, below you're provided a method you can use to caption your own images with the model you've just trained. Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for strange results!)

image_url = 'https://media.4-paws.org/1/e/d/6/1ed6da75afe37d82757142dc7c6633a532f53a7d/VIER%20PFOTEN_2019-03-15_001-2886x1999-1920x1330.jpg'
image_path = tf.keras.utils.get_file(origin=image_url)
image = load_image(image_path)

run_and_show_attention(model, image)
Downloading data from https://media.4-paws.org/1/e/d/6/1ed6da75afe37d82757142dc7c6633a532f53a7d/VIER%20PFOTEN_2019-03-15_001-2886x1999-1920x1330.jpg
133359/133359 [==============================] - 0s 0us/step

Implementing the Score BLEU evaluation

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import sentence_bleu
with open('/content/drive/MyDrive/Data/result.txt', 'r') as file:
    data = file.read()
ref = [data.split()]
print(ref)
test = 'a dog is running'.split()
print('BLEU score for test-> {}'.format(sentence_bleu(ref, test)))
test01 = 'a dog running through the grass'.split()
print('BLEU score for test01-> {}'.format(sentence_bleu(ref, test01)))
[['a', 'dog', 'is', 'running', 'through', 'the', 'grass']]
BLEU score for test-> 0.4723665527410147
BLEU score for test01-> 0.5115078115793242
N-gram evaluation

#def simple_precision(ca, refs, n):
 #   ngrams = make_ngrams(ca, n)
 #   count = 0
 #   for ngram in ngrams:
 #       for ref in refs:
 #           if ngram in make_ngrams(ref, n):
 #               count += 1
 #               break
 #   return count / len(ngrams)
#print(simple_precision(test01, ref, 1))
from nltk.translate.bleu_score import sentence_bleu
with open('/content/drive/MyDrive/Data/result.txt', 'r') as file:
    data = file.read()
ref = [data]
print(ref)
test01 = 'a dog and cat running fastly on the beach'
print('Individual 1-gram: %f' % sentence_bleu(ref, test01, weights=(1, 0, 0, 0)))
print('Individual 2-gram: %f' % sentence_bleu(ref, test01, weights=(0, 1, 0, 0)))
print('Individual 3-gram: %f' % sentence_bleu(ref, test01, weights=(0, 0, 1, 0)))
print('Individual 4-gram: %f' % sentence_bleu(ref, test01, weights=(0, 0, 0, 1)))
['a dog is running through the grass']
Individual 1-gram: 0.609756
Individual 2-gram: 0.450000
Individual 3-gram: 0.358974
Individual 4-gram: 0.289474
This website does not host notebooks, it only renders notebooks available on other websites.

Delivered by Fastly, Rendered by OVHcloud

nbviewer GitHub repository.

nbviewer version: 8b013f7

nbconvert version: 7.2.3

Rendered 4 minutes ago
