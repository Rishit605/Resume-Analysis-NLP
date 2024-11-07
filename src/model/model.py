import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Dense
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
class TextAnalysisModel:
    def __init__(self, vocab_size, embed_dim, max_length, conv_units, dense_units, output, kernels=6, regularizer=0.01, dropout=0.2):
        super(TextAnalysisModel, self).__init__()

        self.conv_units = conv_units
        self.dense_units = dense_units
        self.output = output
        self.kernels = kernels
        self.regularizer = regularizer
        self.dropout = dropout

    def build_model(self):
        model = Sequential([
            layers.Embedding(vocab_size, embed_dim, input_length=max_length),
            layers.Conv1D(self.conv_units, self.kernels, activation='relu', kernel_regularizer=regularizers.l2(self.regularizer)),  # Adding L2 regularization
            layers.GlobalMaxPooling1D(),
            # layers.Dropout(self.dropout),  # Adding dropout to prevent overfitting
            layers.Dense(self.dense_units, activation='relu', kernel_regularizer=l1(self.regularizer)),  # Adding L1 regularization
            layers.Dropout(self.dropout),  # Adding dropout to prevent overfitting
            layers.Dense(self.output, activation='softmax', kernel_regularizer=l2(self.regularizer)),  # Adding L2 regularization
        ])
        return model

## Model Building
class TextAnalysisModel2(Model):
    def __init__(self, vocab_size, embed_dim, max_length, conv_units, dense_units, output, kernels=6, regularizer=0.01, dropout=0.2, embedding_matrix=None):
        super(TextAnalysisModel2, self).__init__()

        # Initialize the embedding layer with weights if provided
        self.embedding = Embedding(vocab_size, embed_dim, input_length=max_length, weights=[embedding_matrix] if embedding_matrix is not None else None)
        self.conv1 = Conv1D(conv_units, kernels, activation='relu', kernel_regularizer=l2(regularizer))  # Adding L2 regularization
        self.global_pool = GlobalMaxPooling1D()
        self.dropout1 = Dropout(dropout)  # First dropout to prevent overfitting
        self.dense1 = Dense(dense_units, activation='relu', kernel_regularizer=l1(regularizer))  # Adding L1 regularization
        self.dropout2 = Dropout(dropout)  # Second dropout to prevent overfitting
        self.dense2 = Dense(output, activation='softmax', kernel_regularizer=l2(regularizer))  # Adding L2 regularization

        if embedding_matrix is not None:
            self.embedding.trainable = False  # Freeze embedding layer if using Word2Vec

    def call(self, inputs, training: bool = False):
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.global_pool(x)

        if training:
            x = self.dropout1(x)

        x = self.dense1(x)

        if training:
            x = self.dropout2(x)

        return self.dense2(x)