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

from evaluate import exact_match_score, f1_score, metric_max_over_ground_truths
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
            W_ques_proj = tf.get_variable("W_ques_proj",shape=(1, output_size_int, output_size_int), 
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_ques_proj = tf.get_variable("b_ques_proj",shape=(output_size_int,), initializer=tf.zeros_initializer())
        
        W_ques_proj = tf.tile(W_ques_proj,[batch_size, 1, 1])
        # # final_questions_encoder_outputs - [batch_size, n+1, output_size]
        final_questions_encoder_outputs = tf.nn.tanh(tf.matmul(questions_encoder_outputs, W_ques_proj) + b_ques_proj)

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
    def __init__(self, output_size, max_iterations=4):
        self.output_size = output_size
        self.max_iterations = max_iterations

    def decode(self, knowledge_rep, batch_size=None):
        """
        Takes in a knowledge representation and output a probability estimation over
        all paragraph tokens on which token should be the start of the answer span, and 
        which should be the end of the answer span.

        args:
            -   knowledge_rep:  representation of the context and question returned by
                                encoder
        return:
            -   start and end of span and prediction score for all the words in the context
        """
        decoder_cell = tf.contrib.rnn.LSTMCell(self.output_size)
        
        # Predicted output and state at each time step
        output_states, preds = [], []
        
        # Get the batch size to get zero state of LSTM 
        if batch_size is None:
            batch_size = tf.shape(knowledge_rep)[0]
        
        cell_state = decoder_cell.zero_state(batch_size, tf.float32)

        max_length = tf.shape(knowledge_rep)[1]

        # Random start and end for the answer span
        init_span = tf.random_uniform(shape=[2,], maxval=max_length, dtype=tf.int32)

        s = tf.tile([init_span[0]],[batch_size])
        e = tf.tile([init_span[1]],[batch_size])

        hs_size = cell_state[1].get_shape().as_list()[1]
        us_size = knowledge_rep.get_shape().as_list()[2]

        # Create Highway Maxout Network class object
        hmn = HMN(hs_size, us_size, batch_size, max_length)
        
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
        
        preds = [alpha, beta]
        
        return  s, e, preds

class QASystem(object):
    def __init__(self, encoder, decoder, context_max_len=None, ques_max_len=None, batch_size=None):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        """
        self.encoder = encoder
        self.decoder = decoder

        # These are None. Earlier value was being passed initially but that makes 
        # things hard while testing as we don't want to pad then.
        self.context_max_len = context_max_len
        self.ques_max_len = ques_max_len
        self.batch_size = batch_size

        self.questions_placeholder = tf.placeholder(shape=(batch_size, self.ques_max_len), dtype=tf.int32)
        self.label_spans_placeholder = tf.placeholder(shape=(batch_size, 2), dtype=tf.int32)
        self.contexts_placeholder = tf.placeholder(shape=(batch_size, self.context_max_len), dtype=tf.int32)
        self.embeddings = tf.placeholder(shape=(encoder.vocab_size, encoder.vocab_dim), dtype=tf.float32)
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
        self.mask_placeholder = tf.placeholder(shape=(batch_size, self.context_max_len), dtype=tf.bool)

        # Setup the computational graph
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.add_training_op(self.loss)

    def setup_system(self):
        """
        Calling encoder, decoder functions to assemble the reading comprehension system.
        """
        knowledge_rep = self.encoder.encode((self.inputs_context, self.inputs_question))
        self.start, self.end, preds = self.decoder.decode(knowledge_rep, self.batch_size)
        self.alpha, self.beta = preds

    def setup_loss(self):
        """
        Loss computation
        """
        with vs.variable_scope("loss"):
            cross_entropy_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_spans_placeholder[:,0],
                                                                             logits=self.alpha)
            cross_entropy_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_spans_placeholder[:,1],
                                                                             logits=self.beta)
            
            # This needs more thinking on how masking can be done.
            # x = tf.boolean_mask(cross_entropy_s, self.mask_placeholder)
            # y = tf.boolean_mask(cross_entropy_e, self.mask_placeholder)
        
            cross_entropy_loss = tf.add(cross_entropy_s, cross_entropy_e)
            self.loss = tf.reduce_mean(cross_entropy_loss)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokensS
        """
        with vs.variable_scope("embeddings"):
            self.inputs_question = tf.nn.embedding_lookup(self.embeddings, self.questions_placeholder)
            self.inputs_context = tf.nn.embedding_lookup(self.embeddings, self.contexts_placeholder)

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(loss)

    def optimize(self, session, train_x, train_y, mask_context, embeddings, dropout=0.5):
        """
        Takes in actual data to optimize your model.
        args:
            session - tensorflow session
            train_x - (context_tokens, question_tokens)
            train_y - answer span (e.g. [2 5])
            embeddings - word embedding for each word in vocabulary
        return:
            loss and train_op
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

    def test(self, session, valid_x, valid_y, embeddings):
        """
        Compute a cost for your validation set and tune your hyperparameters
        according to the validation set performance.
        
        return: Loss on test set
        """
        input_feed = {
            self.questions_placeholder: valid_x[1],
            self.contexts_placeholder: valid_x[0],
            self.label_spans_placeholder: valid_y,
            self.embeddings: embeddings,
            self.dropout_placeholder: 1.0
        }

        output_feed = self.loss

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x, embeddings):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {
            self.questions_placeholder: test_x[1],
            self.contexts_placeholder: test_x[0],
            self.embeddings: embeddings,
            self.dropout_placeholder: 1.0
        }

        output_feed = [self.alpha, self.beta]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x, embeddings):
        
        a_s = []
        a_e = []
        
        for i in range(len(test_x[0])):
            pred = self.decode(session, 
                               (np.reshape(test_x[0][i],[1, len(test_x[0][i])]), 
                                np.reshape(test_x[1][i],[1, len(test_x[1][i])])), 
                               embeddings)
            a_s.append(np.argmax(pred[0]))
            a_e.append(np.argmax(pred[1]))

        return (a_s, a_e)

    def validate(self, sess, valid_dataset, embeddings):
        """
        Iterate through the validation dataset and determine what the validation cost
        is. This method calls self.test() which explicitly calculates validation cost.

        args:
            sess - running tensorflow session
            valid_dataset - ((valid_context_tokens, valid_question_tokens), valid_answer_tokens)
        return:
            valid_cost - validation set loss
        """
        valid_cost = 0.
        for valid_x, valid_y in valid_dataset:
            context = np.reshape(valid_x[0],(1,len(valid_x[0])))
            question = np.reshape(valid_x[1],(1,len(valid_x[1])))
            answer = np.reshape(valid_y,(1,len(valid_y)))
            valid_cost = self.test(sess, (context, question), answer, embeddings)

        return valid_cost

    def evaluate_answer(self, session, val_context_dataset, val_question_dataset, 
                           val_answer_dataset, embeddings, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        args:
            session: session should always be centrally managed in train.py
            dataset: a representation of our data (context, question, answer_span)
            sample: how many examples in dataset we look at
            log: whether we print to std out stream
        
        return:
            (f1, em): f1-score and exact match score
        """
        (val_context_tokens_data, val_context_words_data) = val_context_dataset
        (val_question_tokens_data, val_question_words_data) = val_question_dataset
        (val_answer_tokens_data, val_answer_words_data) = val_answer_dataset
        
        start, end = self.answer(session, (val_context_tokens_data, val_question_tokens_data), embeddings)
        predictions = []
        for i in range(len(start)):
            ans = ""
            for j in range(start[i], end[i]):
                ans += val_context_words_data[i][j] + " "
            ans.strip()
            predictions.append(ans)
        print(len(val_context_tokens_data))
        assert len(predictions) == len(val_answer_words_data),"Shape of predictions and ground truths doesn't match."
        
        exact_match = 0
        f1 = 0

        # Exact Match and F1 Score
        for i in range(len(predictions)):
            exact_match += metric_max_over_ground_truths(exact_match_score, predictions[i], val_answer_words_data[i])
            f1 += metric_max_over_ground_truths(f1_score, predictions[i], val_answer_words_data[i])
        
        total = len(predictions)
        
        em = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        # Log the results
        if log:
            logging.info("F1 Score: {}, EM Score: {}".format(f1, em))

        return f1, em

    def train(self, session, train_dataset, valid_dataset, train_dir, embeddings, batch_size=8, num_epochs=10):
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

        # Number of parameters in your model
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        # Segregate the train data
        inputs, train_span_data, mask_data = train_dataset
        train_context_data, train_question_data = inputs
        inputs_size = len(train_context_data)

        # Segregate the validation data
        (val_context_dataset, val_question_dataset, val_answer_dataset) = valid_dataset
        (val_context_tokens_data, val_context_words_data) = val_context_dataset
        (val_question_tokens_data, val_question_words_data) = val_question_dataset
        (val_answer_tokens_data, val_answer_words_data) = val_answer_dataset

        valid_dataset = zip(zip(val_context_tokens_data, val_question_tokens_data), val_answer_tokens_data)
        mask_context, mask_ques = mask_data

        num_batches = int(inputs_size / batch_size)

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

            print("Training Loss after {0} epochs, {1}".format(i,train_loss))
            
            print("###### Calculating Loss on validation set after epoch {}...".format(i))
            
            valid_loss = self.validate(session, valid_dataset, embeddings)
            
            print("Validation Loss after {0} epochs, {1}".format(i,valid_loss))

            print("###### Evaluating F1 score and EM score...")

            f1, em = self.evaluate_answer(session, 
                                    val_context_dataset, 
                                    val_question_dataset, 
                                    val_answer_dataset, embeddings, log=True)

            print("After {} epochs, F1 Score: {}, EM Score: {}".format(i, f1, em))

            save_time = "{:%Y%m%d_%H%M%S}".format(datetime.now())
            saver.save(session, pjoin(train_dir, save_time + "_model.weights"))