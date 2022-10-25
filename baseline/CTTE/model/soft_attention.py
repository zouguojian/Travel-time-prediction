# -- coding: utf-8 --
import tensorflow as tf

class Attention_box(object):

    @staticmethod
    def soft_attention(query, visualize_attention=False):

        # query is three dim tensor
        # batch x max sequence length x dim { if output is from bi-lstm }
        # other output also should be three dim
        # example if logit is   12 x 10 x 24

        # dim_shape = 24
        dim_shape = query.shape[2]
        reshape_tensor = tf.reshape(query, [-1, dim_shape])
        # reshape : 12 x 10 x 24 ==>  120 x 24

        # 24 x 1
        attention_size = tf.get_variable(name='attention_size',
                                         shape=[dim_shape, 1],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-0.01, 0.01))
        # bias 1
        bias = tf.get_variable(name='bias', shape=[1],
                               dtype=tf.float32,
                               initializer=tf.random_uniform_initializer(-0.01, 0.01))

        # projection without activation
        # ex : 120x24 matmul 24x 1 ==> 120x1
        attention_projection = tf.add(tf.matmul(reshape_tensor, attention_size), bias)
        # reshape . 120x1 ==> 12x10x1 (shape of input )
        output_reshape = tf.reshape(attention_projection, [tf.shape(query)[0], tf.shape(query)[1], -1])
        # softmax over logits 12x10x1
        attention_output = tf.nn.softmax(output_reshape, dim=1)

        # reshape for 2-D for visualization
        attention_visualize = tf.reshape(attention_output,
                                         [tf.shape(query)[0],
                                          tf.shape(query)[1]],
                                         name='Plot')

        attention_projection_output = tf.multiply(attention_output, query)

        Final_output = tf.reduce_sum(attention_projection_output, 1)

        if visualize_attention:

            return {
                'attention_output': attention_projection_output,
                'visualize vector': attention_visualize,
                'reduced_output': Final_output
            }

        else:

            return {
                'attention_output': attention_projection_output,
                'reduced_output': Final_output
            }


# from attention_box import soft_attention
#
# # query is from embedding layer or lstm or cnn logit
#
# attention_output = soft_attention(query, visualize_attention = True )