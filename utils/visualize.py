import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt, ceil
from sklearn.manifold import TSNE


def plot_distribution(data, ax=None, title=None):
    if len(data.shape) == 2:
        data = data.reshape((len(data), ))
    dist = pd.Series(data).value_counts()
    sns.barplot(x=dist.index.values, y=dist.values, ax=ax)
    ax.set(xlabel="Class", ylabel="Number of occurences")
    if title:
        ax.set_title(title)
    
   
def plot_distributions(datasets, titles):
    fig, axes = plt.subplots(1, len(datasets), figsize=(8*len(datasets), 4))
    for i in range(len(axes)):
        plot_distribution(datasets[i], ax=axes[i], title=titles[i])
    plt.subplots_adjust(wspace=0.4)


def show_image(image):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.axis('off')
    plt.show()


def show_image_chunk(images, subfig=None):
    dim = ceil(sqrt(len(images)))
    if subfig:
        axs = subfig.subplots(dim, dim, subplot_kw={'xticks': [], 'yticks': []})
    else:
        _, axs = plt.subplots(dim, dim, figsize=(5, 5), subplot_kw={'xticks': [], 'yticks': []})
    plt.axis('off')
    row, col = 0, 0
    for image in images:
        axs[row, col].imshow(image, cmap=plt.cm.cividis)
        col += 1
        if col == dim:
            row += 1
            col = 0
    
    
def show_image_chunks(image_sets):
    fig = plt.figure(constrained_layout=True, figsize=(5*len(image_sets), 4))
    subfigs = fig.subfigures(1, len(image_sets), wspace=0.4)
    for i in range(len(subfigs)):
        show_image_chunk(image_sets[i], subfig=subfigs[i])


def tsne_plot(data, labels, title=None, ax=None):
    n = pd.Series(labels.reshape((len(labels), ))).nunique()
    transformed_data = TSNE().fit_transform(data)
    sns.scatterplot(
        x=transformed_data[:, 0],
        y=transformed_data[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", n),
        legend="full",
        ax=ax
    )
    if title:
        ax.set_title(title)
    

def tsne_plots(datasets, titles, labels):
    fig, axes = plt.subplots(1, len(datasets), figsize=(8*len(datasets), 4))
    for i in range(len(axes)):
        tsne_plot(datasets[i], labels, title=titles[i], ax=axes[i])
    plt.subplots_adjust(wspace=0.4)
    
    
def plot_logs(title, logs_sets, metric, data_labels, y_label, x_label="Epoch"):
    colors = ['#38abba', '#bf9215', '#3acf61', '#eb4034', '#81a346', '#2d1278']
    color_index = 0

    val_metric = "val_" + metric
    
    fig = plt.figure()
    
    for i in range(len(logs_sets)):
        logs = logs_sets[i]
        data = []
        val_data = []
        
        for log in logs:
            data.append(log[metric])
            val_data.append(log[val_metric])
        
        x_axis = np.arange(len(data))
        plt.plot(x_axis, data, marker='', color=colors[color_index], linewidth=2, label=(data_labels[i]))
        plt.plot(x_axis, val_data, marker='', color=colors[color_index + 1], linewidth=2, label=("Val " + data_labels[i]))
            
        color_index += 2
    
    plt.xlabel(x_label, fontdict={'size': 15})
    plt.ylabel(y_label, fontdict={'size': 15})
    plt.legend(fontsize="large")
    
    
def compare_accuracy(data, labels):
    sns.barplot(x=labels, y=data)