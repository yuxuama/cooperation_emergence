"""animation.py
Animate histogram in order to catch the dynamic of the network
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from analysis import histogram
import numpy as np
import os

class Animation:

    def __init__(self, framerate, save_dir, ostack, parameters):
        
        self.framerate = framerate
        self.ostack = ostack
        self.parameters = parameters
        self.save_dir = save_dir

        # Parse save name
        self.file_radical = self.save_dir + self.parameters["Output directory"].split("/")[-2] + "/"
        os.makedirs(self.file_radical, exist_ok=True)

    
    def set_framerate(self, framerate):
        self.framerate = framerate

    def animate_histogram(self, start, end, step, phenotype, max_trust):
        """Animate histogram"""

        assert phenotype in self.parameters["Strategy distributions"] or phenotype == "Global"

        number_of_frame = (end - start + 1) // step
        rest_time = (1/self.framerate)*1000 # 1000 is for putting it in seconds

        t, _ = self.ostack.resolve(start)        
        bins = np.arange(max_trust + 2)
        n = np.arange(max_trust + 1)
        ph_mean = histogram(t, self.parameters, bins)
        fig, ax = plt.subplots(1, 1)
        _, _, bars = ax.hist(n, bins=bins, weights=ph_mean[phenotype], density=False)
        ax.set(
            xlabel="Trust",
            ylabel="Occurence",
            title="Histogram of {} @ interaction 0".format(phenotype)
        )

        def animate(frame):
            reader = start + frame * step
            t, _ = self.ostack.resolve(reader)
            ph_mean = histogram(t, self.parameters, bins)
            patches = bars.patches
            for i in range(n.size):
                patches[i].set_height(ph_mean[phenotype][i])
            ax.set_title("Histogram of {0} @ interaction {1}".format(phenotype, reader))
            ax.set_ylim([0, np.max(ph_mean[phenotype])+1])

            return bars.patches
        
        ani = FuncAnimation(fig, animate, number_of_frame, interval=rest_time, blit=True, repeat=False)
        ani.save(self.file_radical + phenotype + ".mp4")

        return ani


        
