from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse

import tensorflow as tf
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin
from qa_data import PAD_ID

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 200, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 50, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("dummy_data_dir", "data/squad/dummmy", "Dummy data directory (default ./data/squad/dummy)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def get_data(data_path):
    if tf.gfile.Exists(data_path):
        data = []
        with tf.gfile.GFile(data_path, mode="rb") as f:
            for line in f.readlines():
                data.append([int(x) for x in line.strip().split()])
        return data
    else:
        raise ValueError("Data file %s not found.", data_path)

def pad_sequences(data, max_length, tuples=False):
    """
    Pad the data with PAD_ID which has length less than max_length and 
    discard data with more length.
        
    args:
        -   data: list of sentences
        -   max_length: sequence of length max_length to be maintained
    
    return:
        -   padded_data: data padded with PAD_ID
    """
    if tuples:
        a, b = [], []
        mask_a, mask_b = [], []
        count = 0
        for d in data:
            x, y = d
            #print("Max Length Context {0}, Context Length {1}, Max Length Question {2}, Question Length {3}".format(
                #max_length[0], len(x), max_length[1], len(y)))
            if ((len(x) < max_length[0]) and (len(y) < max_length[1])):
                a.append(x + [PAD_ID]*(max_length[0] - len(x)))
                b.append(y + [PAD_ID]*(max_length[1] - len(y)))
                mask_a.append([True]*len(x) + [False]*(max_length[0] - len(x)))
                mask_b.append([True]*len(y) + [False]*(max_length[1] - len(y)))
        new_data = (a, b)
        mask_data = (mask_a, mask_b)
        print("Count", count)
    else:
        new_data = [d + [PAD_ID]*(max_length - len(d)) for d in data if len(d) < max_length]
        mask_data = [[True]*len(d) + [False]*(max_length - len(d)) for d in data if len(d) < max_length]
    return (new_data, mask_data)

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    train_context_data = get_data(pjoin(FLAGS.data_dir, "train.ids.context"))
    train_question_data = get_data(pjoin(FLAGS.data_dir, "train.ids.question"))
    print("Train Context", len(train_context_data))
    context_max_len = 500
    ques_max_len = 40
    # Pad data with PAD_ID
    inputs, mask_data = pad_sequences(zip(train_context_data, train_question_data), 
                           (context_max_len,ques_max_len), tuples=True)
    print("Input padding done!")
    print("Train Context inputs", len(inputs[0]))
    train_span_data = get_data(pjoin(FLAGS.data_dir, "train.span"))
    print("Train span data loaded!")

    dataset = (inputs, train_span_data, mask_data)

    embed_path = pjoin(FLAGS.data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    vocab_size = len(rev_vocab)
    print("Vocabulary of size {0} initialized!".format(vocab_size))

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, vocab_size=vocab_size)
    decoder = Decoder(output_size=FLAGS.state_size, max_length=context_max_len)

    qa = QASystem(encoder, decoder, context_max_len, ques_max_len)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    embeddings = np.load(embed_path)
    
    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir, embeddings['glove'])

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
