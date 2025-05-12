import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2, l1

# ## MODEL BUILDING
# model = Sequential([
#     layers.Embedding(vocab_size, embed_dim, input_length=max_length),
#     layers.Conv1D(conv_units, kernels, activation='relu', kernel_regularizer=regularizers.l2(regularizer)),  # Adding L2 regularization
#     layers.GlobalMaxPooling1D(),
#     layers.Dropout(dropout),  # Adding dropout to prevent overfitting
#     layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l1(regularizer)),  # Adding L1 regularization
#     layers.Dropout(dropout),  # Adding dropout to prevent overfitting
#     layers.Dense(output, activation='softmax', kernel_regularizer=regularizers.l2(regularizer)),  # Adding L2 regularization
# ])

## MODEL BUILDING
class TextClassifier(Model):
    def __init__(self, vocab_size, embed_dim, num_classes, embedding_matrix=None):
        super(TextClassifier, self).__init__()
        
        # Store these values as instance variables
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.embedding_matrix = embedding_matrix
        
        # Embedding layer with input_length specified
        self.embedding = Embedding(
            vocab_size, 
            embed_dim,
            input_length=500,  # Specify the sequence length
            weights=[embedding_matrix] if embedding_matrix is not None else None,
            trainable=embedding_matrix is None
        )
        
        # Multiple parallel convolution layers
        self.conv1 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv2 = Conv1D(32, 4, activation='relu', padding='same')
        self.conv3 = Conv1D(32, 5, activation='relu', padding='same')
        
        # Pooling layers
        self.pool1 = GlobalMaxPooling1D()
        self.pool2 = GlobalMaxPooling1D()
        self.pool3 = GlobalMaxPooling1D()
        
        # Batch normalization and dropout
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(0.4)
        
        # Dense layers
        self.dense1 = Dense(128, activation='relu')
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(0.4)
        
        self.dense2 = Dense(64, activation='relu')
        self.batch_norm3 = BatchNormalization()
        self.dropout3 = Dropout(0.3)
        
        # Output layer
        self.output_layer = Dense(num_classes, activation='softmax')
        
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs)
        
        # Parallel convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        
        # Pooling
        pool1 = self.pool1(conv1)
        pool2 = self.pool2(conv2)
        pool3 = self.pool3(conv3)
        
        # Concatenate
        concat = tf.keras.layers.concatenate([pool1, pool2, pool3])
        
        # First dense block
        x = self.batch_norm1(concat, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        
        # Second dense block
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        
        # Third dense block
        x = self.batch_norm3(x, training=training)
        x = self.dropout3(x, training=training)
        
        # Output
        return self.output_layer(x)

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'embedding_matrix': None  # Don't save embedding matrix in config
        }
        return config

    @classmethod
    def from_config(cls, config):
        # Extract only the parameters needed for initialization
        init_config = {
            'vocab_size': config.get('vocab_size'),
            'embed_dim': config.get('embed_dim'), 
            'num_classes': config.get('num_classes'),
            'embedding_matrix': None  # Initialize with None, will be loaded separately
        }
        # Only pass parameters that __init__ expects
        return cls(**init_config)

## Model Building
class TextAnalysisModel2(Model):
    def __init__(self, vocab_size, embed_dim, max_length, conv_units, dense_units, output, kernels=6, regularizer=0.01, dropout=0.2, embedding_matrix=None):
        super(TextAnalysisModel2, self).__init__()

        # Initialize the embedding layer with weights if provided
        self.embedding = Embedding(vocab_size, embed_dim, input_length=max_length, weights=[embedding_matrix] if embedding_matrix is not None else None)
        self.conv1 = Conv1D(conv_units, kernels, activation='relu', padding='same')  # Removed L2 Reg
        self.conv2 = Conv1D(conv_units, kernels, activation='relu', padding='same')
        self.conv3 = Conv1D(conv_units, kernels, activation='relu', padding='same')

        self.global_pool1 = GlobalMaxPooling1D()
        self.global_pool2 = GlobalMaxPooling1D()
        self.global_pool3 = GlobalMaxPooling1D()

        self.dropout1 = Dropout(dropout)  # First dropout to prevent overfitting
        self.dense1 = Dense(dense_units, activation='relu', kernel_regularizer=l1(regularizer))  # Adding L1 regularization
        self.dropout2 = Dropout(dropout)  # Second dropout to prevent overfitting
        self.dense2 = Dense(output, activation='softmax', kernel_regularizer=l2(regularizer))  # Adding L2 regularization

        if embedding_matrix is not None:
            self.embedding.trainable = False  # Freeze embedding layer if using Word2Vec

    def call(self, inputs, training: bool = False):
        x = self.embedding(inputs)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.global_pool1(x)
        x = self.global_pool2(x)
        x = self.global_pool3(x)

        if training:
            x = self.dropout1(x)

        x = self.dense1(x)

        if training:
            x = self.dropout2(x)

        return self.dense2(x)