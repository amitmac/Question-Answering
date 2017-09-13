from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from os.path import join as pjoin
from datetime import datetime

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score
from hmn_model import HMN

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, vocab_dim, vocab_size):
        """
        Set the hidden state size, embedding size and vocabulary size

        args:
            -   size: hidden state size
            -   vocab_dim: word embedding size
            -   vocab_size: number of unique token in vocabulary
        """
        self.size = size
        self.vocab_dim = vocab_dim
        self.vocab_size = vocab_size

    def encode(self, inputs, masks=None, encoder_state_input=None):
        """
        In a generalized encode function, pass in inputs, masks, 
        and an initial hidden state input into this function.

        args:
            -   inputs: (questions/contexts - (batch_size, max_length, embed_size))
            -   masks: make sure tf.nn.dynamic_rnn doesn't iterate through masked steps
            -   encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        return: 
            -   encoded representation of your input.
                It can be context-level representation, word-level representation,
                or both.
        """
        inputs_contexts, inputs_questions  = inputs 
        batch_size = tf.shape(inputs_contexts)[0]
        contexts_encoder_cell = tf.contrib.rnn.LSTMCell(self.size)
        
        #context_encoder_outputs - (batch_size x m x self.size)
        with tf.variable_scope("context"):
            contexts_encoder_outputs, contexts_encoder_final_state = tf.nn.dynamic_rnn(
                                                                        cell=contexts_encoder_cell, 
                                                                        inputs=inputs_contexts, 
                                                                        dtype= tf.float32
                                                                    )
        # Add a sentinal vector to allow the model to not attend to any particular word in input
        context_sentinal_vector = tf.Variable(tf.random_uniform(shape=[1 , self.size]))
        context_sentinal_vector_stacked = tf.tile(tf.reshape(context_sentinal_vector,[1,1,-1]),[batch_size, 1, 1])
        contexts_encoder_outputs = tf.concat([contexts_encoder_outputs, context_sentinal_vector_stacked], axis=1)
        # contexts_encoder_outputs (batch_size, m+1, output_size)

        questions_encoder_cell = tf.contrib.rnn.LSTMCell(self.size)
        with tf.variable_scope("questions"):
            questions_encoder_outputs, questions_encoder_final_state = tf.nn.dynamic_rnn(
                                                                        cell=questions_encoder_cell, 
                                                                        inputs=inputs_questions, 
                                                                        dtype= tf.float32
                                                                    )
        a = tf.shape(questions_encoder_outputs)
        batch_size_tensor, max_length_ques_tensor, output_size_tensor = a[0], a[1], a[2]
        batch_size_int, max_length_ques_int, output_size_int  = questions_encoder_outputs.get_shape().as_list()
        
        # Add a sentinal vector to allow the model to not attend to any particular word in input    
        question_sentinal_vector = tf.Variable(tf.random_uniform(shape=(1, self.size)))
        question_sentinal_vector_stacked = tf.tile(tf.reshape(question_sentinal_vector,[1,1,-1]), [batch_size, 1, 1])                       
        questions_encoder_outputs = tf.concat([questions_encoder_outputs, question_sentinal_vector_stacked], axis=1)
        # questions_encoder_outputs (batch_size, n+1, output_size)

        # For variation between question and context encoding space, add a non-linear projection layer
        with tf.variable_scope("encoder"):
            W_ques_proj = tf.get_variable("W_ques_proj",shape=(max_length_ques_int + 1, output_size_int), 
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_ques_proj = tf.get_variable("b_ques_proj",shape=(output_size_int,), initializer=tf.zeros_initializer())
        # # final_questions_encoder_outputs - [batch_size, n+1, output_size]
        final_questions_encoder_outputs = tf.nn.tanh(tf.multiply(W_ques_proj, questions_encoder_outputs) + b_ques_proj)

        # Coattention Encoder
        aff_scores = tf.matmul(contexts_encoder_outputs, tf.transpose(final_questions_encoder_outputs,[0,2,1]))
        # aff_scores - [batch_size, m+1, n+1]

        # Normalize affinity scores across rows and columns to get attention weights for each word in ques and context
        attention_weights_questions = tf.nn.softmax(aff_scores, dim=1)
        attention_weights_contexts = tf.nn.softmax(aff_scores, dim=2)

        attention_context_with_question = tf.matmul(tf.transpose(contexts_encoder_outputs,[0,2,1]), 
                                                    attention_weights_questions) # batch_size x output_size x (n+1)
        attention_question_with_context = tf.transpose(tf.matmul(attention_weights_contexts,
                                                                 final_questions_encoder_outputs),
                                                       [0,2,1]) # batch_size x output_size x (m+1)
        question_to_context_space = tf.matmul(attention_context_with_question, 
                                              tf.transpose(attention_weights_contexts,[0,2,1]))
        # question_to_context_space - (batch_size x output_size x (m+1))
        context_ques_codependent_rep = tf.transpose(tf.concat([attention_question_with_context, 
                                                               question_to_context_space], axis=1),[0,2,1])
        # context_ques_codependent_rep - (batch_size x (m+1) x 2*output_size)

        # Run a Bidirectional LSTM to get the final encoded representation
        new_input = tf.concat([contexts_encoder_outputs, context_ques_codependent_rep], axis=2)
        cell_fw = tf.contrib.rnn.LSTMCell(self.size)
        cell_bw = tf.contrib.rnn.LSTMCell(self.size)
        ((encoder_fw_outputs, encoder_bw_outputs), 
         (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                                             cell_bw=cell_bw,
                                                                                             inputs=new_input,
                                                                                             dtype=tf.float32)
        # concatenate and removing last time step output as that was from sentinal vector not actual word
        final_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), axis=2)
        return final_outputs[:,:-1,:] # batch_size x m x 2*output_size


class Decoder(object):
    def __init__(self, output_size, max_length, max_iterations=4):
        self.output_size = output_size
        self.max_length = max_length    # context maximum length
        self.max_iterations = max_iterations

    def decode(self, knowledge_rep, batch_size=None):
        """
        Takes in a knowledge representation and output a probability estimation over
        all paragraph tokens on which token should be the start of the answer span, and 
        which should be the end of the answer span.

        args:
            -   knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        return:
            -   
        """
        decoder_cell = tf.contrib.rnn.LSTMCell(self.output_size)
        
        # Predicted output and state at each time step
        output_states, preds = [], []
        
        # Get the batch size to get zero state of LSTM 
        if batch_size is None:
            batch_size = tf.shape(knowledge_rep)[0]
        cell_state = decoder_cell.zero_state(batch_size, tf.float32)

        # Random start and end for the answer span
        init_span = np.random.choice(self.max_length, 2)

        s = tf.tile([init_span[0]],[batch_size])
        e = tf.tile([init_span[1]],[batch_size])

        hs_size = cell_state[1].get_shape().as_list()[1]
        us_size = knowledge_rep.get_shape().as_list()[2]

        # Create Highway Maxout Network class object
        hmn = HMN(hs_size, us_size, batch_size, self.max_length)
        
        hmn.initialize_weights("start")
        hmn.initialize_weights("end")
        
        for i in range(self.max_iterations):
            s = tf.to_int32(s)
            e = tf.to_int32(e)
            
            si = tf.stack([tf.range(batch_size), s], axis=1)
            ei = tf.stack([tf.range(batch_size), e], axis=1)

            inp = tf.concat([tf.gather_nd(knowledge_rep, si), 
                             tf.gather_nd(knowledge_rep, ei)], axis=1)
            
            next_output, cell_state = decoder_cell(inp, cell_state)

            alpha = hmn(knowledge_rep, 
                        cell_state[1], 
                        inp, scope="start")
            
            beta = hmn(knowledge_rep, 
                    cell_state[1], 
                    inp, scope="end")

            # alpha and beta - [batch_size x context_max_length] - score for each word in each batch
            s = tf.argmax(alpha, 1)
            e = tf.argmax(beta, 1)
        
        preds = {'alpha':alpha, 'beta':beta}
        
        return  s, e, preds
    
    def check_size(self, knowledge_rep):
        return tf.shape(knowledge_rep)[0]

class QASystem(object):
    def __init__(self, encoder, decoder, context_max_len, ques_max_len, batch_size=None):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder

        self.context_max_len = context_max_len
        self.ques_max_len = ques_max_len
        self.batch_size = batch_size

        self.questions_placeholder = tf.placeholder(shape=(batch_size, self.ques_max_len), dtype=tf.int32)
        self.label_spans_placeholder = tf.placeholder(shape=(batch_size, 2), dtype=tf.int32)
        self.contexts_placeholder = tf.placeholder(shape=(batch_size, self.context_max_len), dtype=tf.int32)
        self.embeddings = tf.placeholder(shape=(encoder.vocab_size, encoder.vocab_dim), dtype=tf.float32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
        self.mask_placeholder = tf.placeholder(shape=(batch_size, self.context_max_len), dtype=tf.bool)

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.train_op = self.add_training_op(self.loss)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder, call various 
        functions inside encoder, decoder here to assemble your reading comprehension system!
        """
        knowledge_rep = self.encoder.encode((self.inputs_context, self.inputs_question))
        self.start, self.end, self.preds = self.decoder.decode(knowledge_rep, self.batch_size)

    def check_(self):
        knowledge_rep = self.encoder.encode((self.inputs_context, self.inputs_question))
        self.b_size = self.decoder.check_size(knowledge_rep)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            cross_entropy_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_spans_placeholder[:,0],
                                                                             logits=self.preds['alpha'])
            cross_entropy_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_spans_placeholder[:,1],
                                                                             logits=self.preds['beta'])
            # print(cross_entropy_s.get_shape().as_list(), self.preds['alpha'].get_shape().as_list(),
            #      self.label_spans_placeholder[:,0].get_shape().as_list())

            # x = tf.boolean_mask(cross_entropy_s, self.mask_placeholder)
            # y = tf.boolean_mask(cross_entropy_e, self.mask_placeholder)
        
            cross_entropy_loss = tf.add(cross_entropy_s, cross_entropy_e)
            self.loss = tf.reduce_mean(cross_entropy_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            self.inputs_question = tf.nn.embedding_lookup(self.embeddings, self.questions_placeholder)
            self.inputs_context = tf.nn.embedding_lookup(self.embeddings, self.contexts_placeholder)

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss)
        return train_op

    def optimize(self, session, train_x, train_y, mask_context, embeddings, dropout=0.5):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {
            self.questions_placeholder: train_x[1],
            self.contexts_placeholder: train_x[0],
            self.embeddings: embeddings,
            self.label_spans_placeholder: train_y,
            self.dropout_placeholder: dropout
        }
        output_feed = [self.loss, self.train_op]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        Compute a cost for your validation set and tune your hyperparameters
        according to the validation set performance.
        
        return: Loss on test set
        """
        input_feed = {
            self.questions_placeholder: valid_x[1],
            self.contexts_placeholder: valid_x[0],
            self.label_spans_placeholder: valid_y,
            self.dropout_placeholder: 1.0
        }

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.questions_placeholder: test_x[1],
            self.contexts_placeholder: test_x[0],
            self.dropout_placeholder: 1.0
        }

        output_feed = [self.pred]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
            valid_cost = self.test(sess, valid_x, valid_y)

        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, train_dir, embeddings, batch_size=8, num_epochs=10):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        #variables_names = [v.name for v in params]
        #values = session.run(variables_names)
        #print(variables_names)
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        inputs, train_span_data, mask_data = dataset
        train_context_data, train_question_data = inputs
        inputs_size = len(train_context_data)

        mask_context, mask_ques = mask_data

        num_batches = int(inputs_size / batch_size)
        #session.run(session.graph.get_tensor_by_name('beta2_power:0').assign(0.99))
        saver = tf.train.Saver()
        for i in range(num_epochs):
            for j in range(num_batches):
                train_context_batch = train_context_data[j*num_batches:(j+1)*num_batches]
                mask_context_batch = mask_context[j*num_batches:(j+1)*num_batches]
                train_question_batch = train_question_data[j*num_batches:(j+1)*num_batches]
                train_span_batch = train_span_data[j*num_batches:(j+1)*num_batches]
                
                train_loss, train_op = self.optimize(session, 
                                                     [train_context_batch, train_question_batch], 
                                                     train_span_batch, mask_context_batch, embeddings)
                
            print("Loss after {0} epochs, {1}".format(i,train_loss))
            save_time = "{:%Y%m%d_%H%M%S}".format(datetime.now())
            saver.save(session, pjoin(train_dir, save_time + "_model.weights"))
            