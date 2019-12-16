import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def smooth(array, interval_length):
    smoothed = np.zeros(len(array))
    for interval_index in range(len(array) // interval_length):
        segment_begin = interval_index * interval_length
        segment_end = (interval_index + 1) * interval_length
        segment = array[segment_begin: segment_end]
        smoothed[segment_begin: segment_end] = np.sum(segment) / interval_length
    return smoothed

def plot_accuracy(base_dir):
    plt.figure()
    # Plot support accuracy
    plt.subplot(1, 2, 1)
    plt.title('Support Accuracy')
    support_accuracy_path = str(Path(base_dir).joinpath('./support_loss.npy'))
    support_accuracy = np.load(support_accuracy_path)
    _, = plt.plot(smooth(support_accuracy, 1))
    plt.ylim((0, 1.05))
    plt.xlim((0, 1000))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Support Accuracy')
    # Plot query accuracy
    plt.subplot(1, 2, 2)
    plt.title('Query Accuracy')
    query_accuracy_path = str(Path(base_dir).joinpath('./query_loss.npy'))
    query_accuracy = np.load(query_accuracy_path)
    _, = plt.plot(smooth(query_accuracy, 1))
    plt.ylim((0, 1.05))
    plt.xlim((0, 1000))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Query Accuracy')
    plt.show()

def main():
    base_dir = os.getcwd()
    plot_accuracy(base_dir)

# This file should be called from within a subdirectory in the output folder, after running main.py
if __name__ == '__main__':
    main()