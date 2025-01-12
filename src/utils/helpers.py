import matplotlib.pyplot as plt

def plot_accuracy(history, accuracy, val_accuracy, save_path: bool = True):
    plt.plot(history.history[accuracy], label='Training Accuracy')
    plt.plot(history.history[val_accuracy], label='Validation Accuracy')
    plt.legend()

    if save_path:
        plt.savefig(r'C:\Projs\COde\ResAnalysis\Resume-Analysis-NLP\src\training\training_plots')
    
    plt.show()

# plot_accuracy(history)