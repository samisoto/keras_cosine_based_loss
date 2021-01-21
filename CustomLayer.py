import tensorflow as tf
import tensorflow_probability as tfp


class CosineLayer(tf.keras.layers.Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(CosineLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    def build(self, input_shape):
        # input_shape = [batch_size,feature_vector_output]
        super(CosineLayer, self).build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.num_classes),
                                      initializer='glorot_normal',
                                      trainable=True)

    @tf.function
    def call(self, inputs, training=None):
        x = tf.math.l2_normalize(inputs, axis=1, name='normalized_x')
        w = tf.math.l2_normalize(self.kernel, axis=0, name='normalized_w')
        cos_t = tf.matmul(x, w, name='cos_t')
        return cos_t


class AdaCos_logits(tf.keras.layers.Layer):

    def build(self, input_shape):
        super(AdaCos_logits, self).build(input_shape)

        x_shape, _ = input_shape
        self.s = tf.Variable(tf.math.sqrt(2.) * tf.math.log(tf.cast(x_shape[-1] - 1, tf.float32)), trainable=False)
        self.correct_cos_mean = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, inputs, training=None):
        cos_t, y_true = inputs

        # y_true.shape = (batch_size,1)
        mask = tf.one_hot(y_true, depth=cos_t.shape[-1], name='one_hot_mask')
        # mask.shape = (batch_size,1,num_classes)
        mask = tf.squeeze(mask, axis=1)
        # mask.shape = (batch_size,num_classes)

        correct_cos_t = tf.reduce_sum(mask * cos_t, axis=1)
        self.correct_cos_mean.assign(tf.reduce_mean(correct_cos_t))

        if training:
            Bavg = (tf.ones_like(mask) - mask) * tf.exp(self.s * cos_t)
            # summarize num_classes
            Bavg = tf.reduce_sum(Bavg, axis=1)
            # average batch
            Bavg = tf.reduce_mean(Bavg, axis=0, name='B_avg')

            cos_med = tfp.stats.percentile(correct_cos_t, q=50, interpolation='midpoint', name='correct_cos_median')
            self.s.assign(tf.math.log(Bavg) / tf.maximum(1 / tf.math.sqrt(2.), cos_med))

        self.add_metric(self.s, name="s")
        self.add_metric(self.correct_cos_mean, name='correct_cos_mean')

        logits = cos_t * self.s

        return logits


class ArcFace_logits(tf.keras.layers.Layer):

    def __init__(self, margin=0.5, scale=32, **kwargs):
        self.margin = margin
        self.s = scale
        super(ArcFace_logits, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'margin': self.margin,
            's': self.s,
        })
        return config

    def build(self, input_shape):
        super(ArcFace_logits, self).build(input_shape)
        self.correct_cos_mean = tf.Variable(0., trainable=False)
        self.cos_m = tf.math.cos(self.margin)
        self.sin_m = tf.math.sin(self.margin)

    @tf.function
    def call(self, inputs, training=None):
        cos_t, y_true = inputs

        # y_true.shape = (batch_size,1)
        mask = tf.one_hot(y_true, depth=cos_t.shape[-1], name='one_hot_mask')
        # mask.shape = (batch_size,1,num_classes)
        mask = tf.squeeze(mask, axis=1)
        # mask.shape = (batch_size,num_classes)

        correct_cos_t = tf.reduce_sum(mask * cos_t, axis=1)
        self.correct_cos_mean.assign(tf.reduce_mean(correct_cos_t))
        self.add_metric(self.correct_cos_mean, name='correct_cos_mean')

        logits = cos_t
        if training:
            sin_t = 1 - tf.square(cos_t)

            #  cos(m+theta) = cosm*cost-sinm*sint
            cos_mt = cos_t * self.cos_m - sin_t * self.sin_m

            # easy_margin
            cos_mt = tf.where(cos_t > 0, cos_mt, cos_t)

            logits = cos_mt * mask + (tf.ones_like(mask) - mask) * cos_t

        logits = self.s * logits

        return logits


class CorrectCosMean(tf.keras.layers.Layer):

    def __init__(self, name="CorrectCosMean", **kwargs):
        super(CorrectCosMean, self).__init__(name=name, **kwargs)
        self.correct_cos_mean = tf.Variable(0., trainable=False)

    @tf.function
    def call(self, inputs):
        cos_t, y_true = inputs
        mask = tf.one_hot(y_true, depth=cos_t.shape[-1], name='one_hot_mask')
        # mask.shape = (batch_size,1,num_classes)
        mask = tf.squeeze(mask, axis=1)
        # mask.shape = (batch_size,num_classes)

        correct_cos_t = tf.reduce_sum(mask * cos_t, axis=1)
        self.correct_cos_mean.assign(tf.reduce_mean(correct_cos_t))
        self.add_metric(self.correct_cos_mean, name='correct_cos_mean')

        return cos_t
