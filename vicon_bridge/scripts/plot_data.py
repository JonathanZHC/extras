import os
import numpy as np
import matplotlib.pyplot as plt

from utils import DataVarIndex, Status, load_data, get_file_path_from_run


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

class Plotter:
    """A class that plots the recorded data."""

    def __init__(self, save_fig=False):
        self.save_fig = save_fig # save generated figure or not      
        self.colors = ['b', 'r']

    def plot_data(self, file_path, plot_indices=None, status=None, plot_arrangement=None):
        # Read the data from the csv file
        raw_data = load_data(file_path)
        observer_file_path = file_path.replace('.csv', '_estimated_data_from_observer.csv')
        estimated_data = load_data(observer_file_path)

        if status is not None:
            # Only plot the data that matches the status
            index = np.where(raw_data[:, DataVarIndex.STATUS] == status.value)
            raw_data = raw_data[index]
            estimated_data = estimated_data[index]
        else:
            max_rows = estimated_data.shape[0]
            raw_data = raw_data[:max_rows, :]

        # Subtract the start time from the time values
        start_time = raw_data[0, DataVarIndex.TIME]
        raw_data[:, DataVarIndex.TIME] -= start_time
        
        # Plot the data      
        num_plots = len(plot_indices) # Total number of plots

        if plot_arrangement:
            if len(plot_arrangement) != 2:
                raise ValueError("plot_arrangement must be a tuple of 2 integers")
            num_rows, num_cols = plot_arrangement
            if num_rows * num_cols <= num_plots:
                raise ValueError("plot_arrangement must have enough subplots to plot all indices")
        else:
            # Create a figure with the number of subplots equal to the number of indices
            # Find closest non-prime number to the number of plots
            if num_plots > 5:
                num_plots_non_prime = num_plots
                while is_prime(num_plots_non_prime):
                    num_plots_non_prime += 1
                
                num_plots = num_plots_non_prime
            
            # Find the closest factors of the number of plots
            for i in range(int(num_plots ** 0.5), 0, -1):
                if num_plots % i == 0:
                    num_cols = i
                    break

            num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 9))

        for plot_id, i in enumerate(plot_indices):
            
            if isinstance(axs, np.ndarray):
                if len(axs.shape) == 1:
                    axs = axs.reshape(1, -1)
                ax = axs[plot_id // num_cols, plot_id % num_cols] if num_rows > 1 else axs[plot_id % num_cols]
            else:
                ax = axs
            if isinstance(i, tuple):
                '''
                if i[1] == "dt":
                    # Plot the actual values
                    dt = np.mean(np.diff(raw_data[:, DataVarIndex.TIME]))
                    ax.plot(raw_data[:-1, DataVarIndex.TIME], np.diff(raw_data[:, i[0]]) / dt, '*-', label=f"d{i[0].name}/ dt", color=color)
                    # set axis labels
                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel(f"d{i[0].name}/dt")
                else:
                    # Check if the desired value is available
                    if i[0] in self.match_desired and i[1] in self.match_desired:
                        # Plot the desired values
                        ax.plot(raw_data[:, self.match_desired[i[0]]], raw_data[:, self.match_desired[i[1]]], "--", color=next_color, 
                                label=self.match_desired[i[0]].name + " vs " + self.match_desired[i[1]].name)
                    
                    # Plot the actual values
                    ax.plot(raw_data[:, i[0]], raw_data[:, i[1]], label=f"{i[0].name} vs {i[1].name}", color=color)
                    
                    # set axis labels
                    ax.set_xlabel(i[0].name)
                    ax.set_ylabel(i[1].name)
                '''

            else:
                # Plot the actual values
                ax.plot(raw_data[:, DataVarIndex.TIME], raw_data[:, i], "--", label='simple estimator', color=self.colors[1], linewidth=1)
                ax.plot(raw_data[:, DataVarIndex.TIME], estimated_data[:, i], color=self.colors[0], label='EKF', linewidth=1.5)
                    
                # set axis labels
                ax.set_xlabel("Time [s]")
                ax.set_ylabel(i.name)
            
            # set the legend
            ax.legend()

        # Save the figure
        if self.save_fig:
            fig_name = os.path.splitext(file_path)[0] + ".png"
            ax.savefig(fig_name)

        plt.show()


if __name__ == "__main__":
    wandb_project = "test"
    # Plot the entire trajectory or just the tracking part
    status = None #status = Status.TRACK_TRAJ

    # Specify the indices to be plotted
    plot_indices = [#(DataVarIndex.POS_X, DataVarIndex.POS_Y), 
                    #(DataVarIndex.POS_Y, DataVarIndex.POS_Z), 
                    #(DataVarIndex.POS_X, DataVarIndex.POS_Z), 
                    #DataVarIndex.POS_X, 
                    #DataVarIndex.POS_Y,
                    #DataVarIndex.POS_Z,
                    DataVarIndex.VEL_X,
                    DataVarIndex.VEL_Y,
                    DataVarIndex.VEL_Z,
                    DataVarIndex.ROLL,
                    DataVarIndex.PITCH,    
                    DataVarIndex.YAW,                
                    #DataVarIndex.CMD_THRUST,
                    #DataVarIndex.ROLL_RATE,
                    #DataVarIndex.YAW_RATE,
                    #DataVarIndex.PITCH_RATE,
                    ] 

    # Specify the data by setting the file_path
    raw_file_path = '/home/haocheng/Experiments/figure_8/data_20240930_151508.csv'

    # Start plotting
    plotter = Plotter(save_fig=False) 
    print("Plotting data from: ", raw_file_path)
    plotter.plot_data(raw_file_path, plot_indices=plot_indices, status=status)
