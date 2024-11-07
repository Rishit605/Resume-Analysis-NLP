import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()

# plot_accuracy(history)