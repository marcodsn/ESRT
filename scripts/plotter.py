import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_dir = 'plots'


class Plot:
    def __init__(self, title, height, width, y_steps, x_steps, y_label, x_label, smoothing_factor=0.0):
        self.title = title
        self.height = height
        self.width = width
        self.y_steps = y_steps
        self.x_steps = x_steps
        self.y_label = y_label
        self.x_label = x_label
        self.smoothing_factor = smoothing_factor
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlabel(x_label)

        # Set aspect ratio
        self.fig.set_size_inches(width / 100, height / 100)

    def add(self, x_data, y_data, label, color, linestyle='-'):
        if self.smoothing_factor > 0:
            smoothed_data = self.smooth_data(y_data, self.smoothing_factor)
            self.ax.plot(x_data, smoothed_data, label=label, linestyle=linestyle, color=color, alpha=1.0)
            self.ax.plot(x_data, y_data, color=color, alpha=0.4, linestyle=linestyle)
        else:
            self.ax.plot(x_data, y_data, label=label, color=color, linestyle=linestyle)
        self.ax.legend()
        self.ax.grid()
        self.ax.set_xticks(np.arange(min(x_data), max(x_data) + 1, self.x_steps))

    def save(self, filename):
        # self.fig.savefig(filename, dpi=300)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        self.fig.savefig(os.path.join(plot_dir, filename), dpi=300)
        plt.show()

    @staticmethod
    def smooth_data(data, smoothing_factor):
        smoothed_data = []
        for i in range(len(data)):
            if i == 0:
                smoothed_data.append(data[i])
            else:
                smoothed_data.append(smoothed_data[-1] * (1 - smoothing_factor) + data[i] * smoothing_factor)
        return smoothed_data


def read_data_from_csv(csv_file, scale=3, banned_suffixes=None):
    if banned_suffixes is None:
        banned_suffixes = ['__MIN', '__MAX']
    df = pd.read_csv(csv_file)
    x_labels = df['Step'].tolist()

    y_data = []
    for col in df.columns:
        if not any([col.endswith(suffix) for suffix in banned_suffixes]):
            y_data.append(df[col].tolist())
    return x_labels, y_data


x = np.linspace(0, 200, num=200, dtype=int)
# FX4, FX3, DX4, DX3
ssim_data = read_data_from_csv('SSIM.csv')[1][1:]
psnr_data = read_data_from_csv('PSNR.csv')[1][1:]
training_loss_data = read_data_from_csv('training_loss.csv')[1][1:]
set5_loss_data = read_data_from_csv('Set5_loss.csv')[1][1:]

height = 450
width = 800
smoothing_factor = 0.3

plot = Plot(title='DIV2K X3 - Loss', height=height, width=width, y_steps=None, x_steps=50, y_label='Loss', x_label='Epochs',
            smoothing_factor=smoothing_factor)
plot.add(x, training_loss_data[3], label='Training', color='green')
plot.add(x, set5_loss_data[3], label='Set5', color='green', linestyle='dashed')
plot.save('DX3_loss.png')

plot2 = Plot(title='Flickr-8k X3 - Loss', height=height, width=width, y_steps=None, x_steps=50, y_label='Loss', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot2.add(x, training_loss_data[1], label='Training', color='purple')
plot2.add(x, set5_loss_data[1], label='Set5', color='purple', linestyle='dashed')
plot2.save('FX3_loss.png')

plot3 = Plot(title='X3 - Loss', height=height, width=width, y_steps=None, x_steps=50, y_label='Loss', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot3.add(x, training_loss_data[3], label='DIV2K - Training', color='green')
plot3.add(x, set5_loss_data[3], label='DIV2K - Set5', color='green', linestyle='dashed')
plot3.add(x, training_loss_data[1], label='Flickr-8k - Training', color='purple')
plot3.add(x, set5_loss_data[1], label='Flickr-8k - Set5', color='purple', linestyle='dashed')
plot3.save('X3_loss.png')

plot4 = Plot(title='X3 - PSNR', height=height, width=width, y_steps=None, x_steps=50, y_label='PSNR', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot4.add(x, psnr_data[3], label='DIV2K', color='green')
plot4.add(x, psnr_data[1], label='Flickr-8k', color='purple')
plot4.save('X3_psnr.png')

plot5 = Plot(title='X3 - SSIM', height=height, width=width, y_steps=None, x_steps=50, y_label='SSIM', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot5.add(x, ssim_data[3], label='DIV2K', color='green')
plot5.add(x, ssim_data[1], label='Flickr-8k', color='purple')
plot5.save('X3_ssim.png')

plot6 = Plot(title='X4 - Loss', height=height, width=width, y_steps=None, x_steps=50, y_label='Loss', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot6.add(x, training_loss_data[2], label='DIV2K - Training', color='green')
plot6.add(x, set5_loss_data[2], label='DIV2K - Set5', color='green', linestyle='dashed')
plot6.add(x, training_loss_data[0], label='Flickr-8k - Training', color='purple')
plot6.add(x, set5_loss_data[0], label='Flickr-8k - Set5', color='purple', linestyle='dashed')
plot6.save('X4_loss.png')

plot7 = Plot(title='X4 - PSNR', height=height, width=width, y_steps=None, x_steps=50, y_label='PSNR', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot7.add(x, psnr_data[2], label='DIV2K', color='green')
plot7.add(x, psnr_data[0], label='Flickr-8k', color='purple')
plot7.save('X4_psnr.png')

plot8 = Plot(title='X4 - SSIM', height=height, width=width, y_steps=None, x_steps=50, y_label='SSIM', x_label='Epochs',
             smoothing_factor=smoothing_factor)
plot8.add(x, ssim_data[2], label='DIV2K', color='green')
plot8.add(x, ssim_data[0], label='Flickr-8k', color='purple')
plot8.save('X4_ssim.png')
