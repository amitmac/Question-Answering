"""
    Highway Maxout Network model

    -   For an effective way to pool across multiple model variations
"""
import tensorflow as tf

class HMN():
    def __init__(self, hs_size, us_size, batch_size, max_length, max_pool_size=16):
        self.hs_size = hs_size # 200
        self.us_size = us_size # 400
        self.max_pool_size = max_pool_size
        self.batch_size = batch_size
        self.max_length = max_length

    def __call__(self, inputs, hidden_state, previous_outputs, scope):
        """
        args:
            -   inputs: batch_size x max_length x output_size
            -   hidden_state: batch_size x hs_size
            -   previous_outputs: batch_size x 2*us_size
        return:
            -   score: batch_size x max_length (score for each word for each data in batch)
        """
        with tf.variable_scope(scope, reuse=True):
            W_d = tf.get_variable("W_d")
            W_1 = tf.get_variable("W_1")
            b_1 = tf.get_variable("b_1")
            W_2 = tf.get_variable("W_2")
            b_2 = tf.get_variable("b_2")
            W_3 = tf.get_variable("W_3")
            b_3 = tf.get_variable("b_3")
        
        inp0 = tf.concat([hidden_state, previous_outputs], axis=1) # batch_size x (hs_size + 2*us_size)
        r = tf.nn.tanh(tf.matmul(inp0, W_d)) # batch_size x hs_size
        # r - [batch_size x l]
        # tile r and reshape to concatenate with inputs
        r = tf.transpose(tf.reshape(tf.tile(r, [self.max_length,1]),[self.max_length, -1 , self.hs_size]),[1,0,2]) 
        inp1 = tf.concat([inputs, r], 2) # shape - [batch_size  x max_length x 3l]

        # change inp1 to 2d matrix [batch_size*max_length x 3l] for ease in operations
        inp1 = tf.reshape(inp1, [self.batch_size*self.max_length, -1])

        #inp1 = tf.tile(tf.reshape(inp1, [1, -1, self.batch_size*self.max_length]),[self.max_pool_size, 1, 1])
        #b_1 = tf.tile(tf.reshape(b_1, [1, self.hs_size, self.max_pool_size]),[self.batch_size*self.max_length, 1, 1])
        m_t1_pre = tf.reshape(tf.matmul(inp1, W_1) + b_1, [-1, self.max_pool_size, self.hs_size])
        m_t1 = tf.reduce_max(m_t1_pre, axis=1)
        # [batch_size*max_length x l]

        #inp2 = tf.tile(tf.reshape(m_t1, [1, self.hs_size, self.max_length*self.batch_size]),[self.max_pool_size, 1, 1])
        #b_2 = tf.tile(tf.reshape(b_2, [1, self.hs_size, self.max_pool_size]),[self.batch_size*self.max_length, 1, 1])
        m_t2_pre = tf.reshape(tf.matmul(m_t1, W_2) + b_2, [-1, self.max_pool_size, self.hs_size])
        m_t2 = tf.reduce_max(m_t2_pre, axis=1)
        # [batch_size*max_length x l]

        inp3 = tf.concat([m_t1, m_t2], axis=1)

        output_ = tf.reduce_max(tf.matmul(inp3, W_3) + b_3, axis=1)
        output = tf.reshape(output_,[self.batch_size, self.max_length])

        return output

    def initialize_weights(self, scope):
        with tf.variable_scope(scope) as vscope:
            W_d = tf.get_variable("W_d", shape=(self.hs_size+2*self.us_size, self.hs_size), 
                                  initializer=tf.contrib.layers.xavier_initializer())
        
            W_1 = tf.get_variable("W_1", shape=(self.us_size+self.hs_size, self.max_pool_size*self.hs_size), 
                                  initializer=tf.contrib.layers.xavier_initializer()) # (p x l x 3l)
            b_1 = tf.get_variable("b_1", shape=(self.hs_size*self.max_pool_size))
        
            W_2 = tf.get_variable("W_2", shape=( self.hs_size, self.max_pool_size*self.hs_size), 
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_2 = tf.get_variable("b_2", shape=(self.hs_size*self.max_pool_size))
        
            W_3 = tf.get_variable("W_3", shape=(2*self.hs_size, self.max_pool_size), 
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_3 = tf.get_variable("b_3", shape=(self.max_pool_size))

            vscope.reuse_variables()

        return True