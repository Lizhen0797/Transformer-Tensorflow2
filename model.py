import tensorflow as tf
from tensorflow import keras

'''
    获取sin和cos中的值
    @params pos 位置
    @params i i
    @params d_model d_model, 一个超参数，保证通过i的位置来保证2i/d_model在0-1之间，保证sin与cos之间的相互表达
    @return sin及cos的值
'''


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float(d_model))
    return pos * angle_rates


'''
    获取指定大小的position_encoding
    @params position 可以理解为seq_len
    @params d_model 可以理解为position_encoding的纬度大小
    @return Positional Encoding metrix
'''


def positional_encoding(position, d_model):
    # 后面两个参数是转换为2维矩阵,第一个为[position,1] 第二个为[1, d_model]
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # 从0开始到最后:,步长为2 获得sin
    sines = np.sin(angle_rads[:, 0::2])
    # 从1开始到最后:,步长为2 获得cos
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


'''
    把seq中的padding提取出来，此处的pad为0
    @params 序列
    @return 序列中的padding
'''


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :

                                          '''
                                             创建哪些是不需要看的矩阵
                                             @param 矩阵横纵坐标
                                             @return 矩阵
                                         '''


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


'''
    点乘Attention
    @params q query [batch_size, seq_len_q, depth] depth是否可以认为是feature_size？
    @params k key [batch_size, seq_len_k, depth]
    @params v value [batch_size, seq_len_v, depth]

    @return output 经过scaled_dot_product_attention得到的结果 shape [batch_size,seq_len, seq_len_k/d_model(划分特征大小)]
    @return attetnion_weights 注意力大小
'''


def scaled_dot_product_attention(q, k, v, mask):
    # result [batch_size,seq_len, seq_len_k]
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


'''
    多头注意力层,多头注意力是指将特征纬度分为多个纬度，从而使得更好获取局部特征
    @param raw_feature 原始特征纬度
    @param num_heads 将raw_feature分成的纬度数量，也是多头注意力的头数
'''


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, raw_feature, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.raw_feature = raw_feature

        assert raw_feature % self.num_heads == 0
        self.depth = raw_feature // self.num_heads

        self.wq = tf.keras.layers.Dense(raw_feature)
        self.wk = tf.keras.layers.Dense(raw_feature)
        self.wv = tf.keras.layers.Dense(raw_feature)

        self.dense = tf.keras.layers.Dense(raw_feature)

    def split_heads(self, x, batch_size):
        # retrun [batch_size, num_heads, seq_len, depth/划分的特征大小]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.raw_feature))

        # (batch_size, seq_len_v/(seq_len), raw_feature/总特征大小]
        output = self.dense(concat_attention)

        return output, attention_weights


'''
    前向传播网络
    @param d_model 第二层隐藏单元数
    @param dff 第一层隐藏单元数

    @return tf.keras.Sequential 
'''


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


'''
    Encoder层
    @param raw_feature 原始特征数
    @param num_heads 多头注意力头数
    @param dff 前向传播网络中间层层数
    @param dropout_rate dropout_rate 
'''


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, raw_feature, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(raw_feature, num_heads)
        self.ffn = point_wise_feed_forward_network(raw_feature, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # 获得多头注意力的输出
        attn_output, _ = self.mha(x, x, x, mask)
        # 输出Dropout
        attn_output = self.dropout1(attn_output, training=training)
        # 残差结构想家
        out1 = self.layernorm1(x + attn_output)
        # 前向传播
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 前向传播的残差
        out2 = self.layernorm1(out1 + ffn_output)

        return out2


'''
    TransformerEncoder,在Transformer翻译模型中Encoder主要是用来捕获输入信息，Decoder主要来捕获输出信息，因此，
    在文本分类这种类似分类中，并不需要Decoder。
    @param num_layers Encoder_layers的层数，这里我可以对应LSTM的层数
    @param raw_feature 特征纬数
    @param num_head 多头注意力的头数
    @param dff 前向传播神经网络的中间层层数
    @param seq_len 序列长度
    @param dropout_rate dropout的dropout比率
'''


class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, raw_feature, num_heads, dff, seq_len, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.raw_feature = raw_feature
        self.num_layers = num_layers
        self.positional_encoding = positional_encoding(seq_len, raw_feature)
        self.enc_layers = [EncoderLayer(raw_feature, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.dense = tf.keras.layers.Dense(1, activation='softmax')

    def call(self, x, training=True, mask=None):
        seq_len = tf.shape(x)[1]
        # 自身加上positional encoding
        x += self.positional_encoding[:, :seq_len, :]

        # 输入前判断是否dropout
        x = self.dropout(x, training=training)

        # encoderlayer多次 输出[batch_size, seq_len, raw_feature]
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        '''
        因输出的还是为[Batch_size,Seq_len,Raw_feature]大小的矩阵，只是通过注意力模型将其累加，所以此时有多种做法，
        这里的做法是特征纬度每一维最大，最后得到[Batch_size,raw_feature]
        还有另一种想发是输入LSTM
        '''
        x = tf.reduce_sum(x, axis=1)
        x = self.dense(x)
        return x