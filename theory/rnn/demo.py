import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = "http://mattmahoney.net/dc/"

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename,_ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename

# 数据下载
filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename='text8.zip')
# 数据单词总量统计
print('Data size', len(words))

vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size -1))
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return  data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)

del words
# 最常见词统计
print('Most common words (+UNK)', count[:5])
#
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:18]])

data_index = 0
def generate_batch(batch_size, num_skips, skip_windows):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2* skip_windows
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2* skip_windows +1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append((data[data_index]))
        data_index +=1

    for i in range(batch_size // num_skips):
        target = skip_windows
        targets_to_avoid = [ skip_windows ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span -1)
            targets_to_avoid.append(target)
            batch[ i * num_skips + j] = buffer[skip_windows]
            labels[i * num_skips + j, 0] =  buffer[target]
        buffer.append(data[data_index])
        data_index += 1
    return batch,labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_windows=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0],
          reverse_dictionary[labels[i, 0]])

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips =2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.in32, shape=[batch_size])
    train_labels = tf.placeholder(tf.in32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size))
    )

    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=train_labels,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


init = tf.global_variables_initializer()