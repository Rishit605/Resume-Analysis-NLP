import matplotlib.pyplot as plt

def plot_accuracy(history, accuracy, val_accuracy):
    plt.plot(history.history[accuracy], label='Training Accuracy')
    plt.plot(history.history[val_accuracy], label='Validation Accuracy')
    plt.legend()
    plt.show()

# plot_accuracy(history)