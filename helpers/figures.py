import matplotlib.pyplot as plt


# plt.plot(pd.DataFrame(fit_stats.history))

def plot_hist(dist_keys):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 2)
    plt.plot(dist_keys.history['accuracy'], label='Training Accuracy')
    plt.plot(dist_keys.history['val_accuracy'], label='Validation Accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim(0)), 2])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('epoch')

    plt.subplot(2, 1, 1)
    plt.plot(dist_keys.history['loss'], label='Training Loss')
    plt.plot(dist_keys.history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.ylim([min(plt.ylim()), 2])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
