import tensorflow as tf
import tensorflow_addons as tfa


class PatchEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.projection = tf.keras.layers.Conv3D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)
        self.num_patches = num_patches

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        embeddings = self.projection(inputs, training=training)
        embeddings = tf.reshape(tensor=embeddings, shape=(tf.shape(inputs)[0], self.num_patches, -1))
        return embeddings


class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        num_patches = (
            (input_shape[1] // self.patch_size)
            * (input_shape[2] // self.patch_size)
            * (input_shape[3] // self.patch_size)
        )
        self.patch_embeddings = PatchEmbeddings(
            self.patch_size, self.hidden_size, num_patches=num_patches, name="patch_embeddings"
        )
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, self.hidden_size),
            trainable=True,
            name="position_embeddings",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        embeddings = self.patch_embeddings(inputs, training=training)
        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)

        return embeddings


class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def build(self, input_shape):
        super().build(input_shape)
        # TODO YONIGO: tf doc says to do this  ¯\_(ツ)_/¯
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores=True)
        return weights


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        outputs = []
        for block in self.blocks:
            x = block(x, training=training)
            outputs.append(x)
        x = self.norm(x)
        return x, outputs


class DoubleYellowBlock(tf.keras.Model):
    def __init__(self, filters, use_bias=False):
        super().__init__()
        self.block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv3D(filters, kernel_size=3, padding="same", use_bias=use_bias),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv3D(filters, kernel_size=3, padding="same", use_bias=use_bias),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.block(inputs, training=training)


class BlueBlock(tf.keras.Model):
    def __init__(self, filters, use_bias=False):
        super().__init__()
        self.block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=2, use_bias=use_bias),
                tf.keras.layers.Conv3D(filters, kernel_size=3, padding="same", use_bias=use_bias),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        return self.block(inputs, training=training)


class DecoderBlock(tf.keras.Model):
    def __init__(self, filters, use_bias=False):
        super().__init__()
        self.green_block = tf.keras.layers.Conv3DTranspose(filters, kernel_size=2, strides=2, use_bias=use_bias)
        self.block = DoubleYellowBlock(filters=filters, use_bias=use_bias)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x, skip = inputs
        x = self.green_block(x)
        x = tf.concat([x, skip], axis=-1)
        return self.block(x, training=training)


class UNETR(tf.keras.Model):
    def __init__(
        self,
        feature_size,
        output_channels,
        patch_size=16,
        vit_hidden_size=768,
        vit_depth=12,
        attention_heads=12,
        mlp_dim=3072,
    ):
        super().__init__()
        self.vit = VisionTransformer(
            patch_size=patch_size,
            hidden_size=vit_hidden_size,
            depth=vit_depth,
            num_heads=attention_heads,
            mlp_dim=mlp_dim,
        )
        self.patch_size = patch_size

        self.encoder1 = DoubleYellowBlock(feature_size)
        self.encoder2 = tf.keras.Sequential(
            [BlueBlock(feature_size * 2), BlueBlock(feature_size * 2), BlueBlock(feature_size * 2)]
        )
        self.encoder3 = tf.keras.Sequential([BlueBlock(feature_size * 4), BlueBlock(feature_size * 4)])
        self.encoder4 = tf.keras.Sequential([BlueBlock(feature_size * 8)])

        self.decoder4 = DecoderBlock(feature_size * 8)
        self.decoder3 = DecoderBlock(feature_size * 4)
        self.decoder2 = DecoderBlock(feature_size * 2)
        self.decoder1 = DecoderBlock(feature_size * 1)

        self.out = tf.keras.Sequential(
            [
                DoubleYellowBlock(feature_size),
                DoubleYellowBlock(feature_size),
                tf.keras.layers.Conv3D(filters=output_channels, kernel_size=1),
            ]
        )

    def build(self, input_shape):
        self.num_patches_dim = input_shape[1] // self.patch_size
        assert input_shape[2] // self.patch_size == self.num_patches_dim
        assert input_shape[3] // self.patch_size == self.num_patches_dim

    @staticmethod
    def reshape3D(inputs, num_patches_dim):
        bs = tf.shape(inputs)[0]
        hidden_size = tf.shape(inputs)[-1]
        return tf.reshape(inputs, (bs, num_patches_dim, num_patches_dim, num_patches_dim, hidden_size))

    def call(self, inputs, training):
        x, zs = self.vit(inputs, training)
        z3, z6, z9 = zs[3], zs[6], zs[9]
        enc1 = self.encoder1(inputs)  # (bs,H,W,D,features_size)
        enc2 = self.encoder2(UNETR.reshape3D(z3, self.num_patches_dim))
        enc3 = self.encoder3(UNETR.reshape3D(z6, self.num_patches_dim))
        enc4 = self.encoder4(UNETR.reshape3D(z9, self.num_patches_dim))

        dec4 = self.decoder4([UNETR.reshape3D(x, self.num_patches_dim), enc4])
        dec3 = self.decoder3([dec4, enc3])
        dec2 = self.decoder2([dec3, enc2])
        dec1 = self.decoder1([dec2, enc1])
        out = self.out(dec1)
        return out
