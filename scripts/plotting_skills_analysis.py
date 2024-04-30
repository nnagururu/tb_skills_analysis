import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits import mplot3d

class StrokeMetricsVisualizer:
    """
    A class to visualize stroke metrics for a given experiment.

    Attributes:
    ------------
        - metrics_dict (dict): Dictionary containing the stroke metrics.
        - bucket_assignments (dict): Dictionary containing the bucket assignments for each metric.
        - metrics_dict2 (dict): Dictionary containing the stroke metrics for a second experiment.
        - bucket_assignments2 (dict): Dictionary containing the bucket assignments for each metric for the second experiment.
        - plot_previous_bucket (bool): If True, plots the previous bucket in addition to the current bucket.
        - metrics_to_plot (list): List of metrics to plot.
        - xlabels (dict): Dictionary mapping metrics to their x-axis labels.
        - titles (dict): Dictionary mapping metrics to their plot titles.
        - num_buckets (int): The number of buckets in the experiment.
        - available_metrics_to_plot (list): List of metrics available to plot.

    Methods:
    ------------
        - init_filtered_dicts(metrics_dict, bucket_dict, metrics_to_plot): Initializes filtered dictionaries for metrics.
        - remove_outliers(data, method='std'): Removes outliers from the data using standard deviation or IQR.
        - precompute_bin_edges(num_bins=15): Precomputes bin edges for histograms.
        - find_max_frequency_per_metric(bin_edges_dict): Finds the maximum frequency for each metric.
        - plot_bucket_data(bucket_index, fig, axes, bin_edges_dict, max_freq_per_metric): Plots the data for a given bucket.
        - _plot_bar(ax, data, bin_edges, bar_width, offset, color, alpha, label): Plots a bar chart for a given metric.
        - interactive_plot_buckets(num_bins=15): Plots an interactive visualization of the stroke metrics.
    
    """
    def __init__(self, metrics_dict, bucket_dict, metrics_dict2=None, bucket_dict2=None, plot_previous_bucket=False, 
                 metrics_to_plot = ['length', 'velocity', 'acceleration', 'vxls_removed',
                                    'curvature', 'force', 'angle_wrt_bone', 'angle_wrt_camera']):
        """
        Initializes the StrokeMetricsVisualizer object.

        Parameters: Class attributes as above
        """
        
        # Subset only common keys between two metrics_dict if specified
        if metrics_dict2 is not None:
            common_keys = set(metrics_dict.keys()).intersection(metrics_dict2.keys())
            metrics_dict = {key: metrics_dict[key] for key in common_keys}
            metrics_dict2 = {key: metrics_dict2[key] for key in common_keys}

        # Instantiate filtered dictionaries
        self.metrics_dict, self.bucket_assignments = self.init_filtered_dicts(metrics_dict, bucket_dict, metrics_to_plot)
        
        if metrics_dict2 is not None:
            self.metrics_dict2, self.bucket_assignments2 = self.init_filtered_dicts(metrics_dict2, bucket_dict2, metrics_to_plot)
        else:
            self.metrics_dict2 = None
            self.bucket_assignments2 = None

        # Dicts for labels
        self.xlabels = {'length': 'length (m)', 'velocity': 'velocity (m/s)', 'acceleration': 'acceleration (m/s^2)',
                        'jerk': 'jerk (m/s^3)', 'vxls_removed': 'voxels removed (unitless)','curvature': 'curvature (unitless)',
                        'angle_wrt_bone': 'angle (degrees)', 'angle_wrt_camera': 'angle (degrees)', 'force': 'force (N)'}
        self.titles = {'length': 'Stroke Length', 'velocity': 'Stroke Velocity', 'acceleration': 'Stroke Acceleration',
                       'jerk': 'Stroke Jerk', 'curvature': 'Stroke Curvature', 'angle_wrt_bone': 'Drill Angle w.r.t. Bone',
                       'angle_wrt_camera': 'Drill Angle w.r.t Camera', 'force': 'Stroke Force', 'vxls_removed': 'Voxels Removed per Stroke'}
        
        self.num_buckets = max(bucket_dict['bucket_assignments']) + 1
        self.plot_previous_bucket = plot_previous_bucket
        self.available_metrics_to_plot = [metric for metric in metrics_to_plot if metric in self.metrics_dict]


    def init_filtered_dicts(self, metrics_dict, bucket_dict, metrics_to_plot):
        """
        Initializes filtered dictionaries for metrics and bucket assignments, where outliers are filtered out. Additionally,
        removes metrics not in metrics_to_plot

            Parameters:
                metrics_dict (dict): Dictionary containing the stroke metrics.
                bucket_dict (dict): Dictionary containing the bucket assignments for each metric.
                metrics_to_plot (list): List of metrics to plot.
            
            Returns: 
                filtered_dict (dict): Dictionary containing the filtered stroke metrics.
                filtered_bucket_assgn (dict): Dictionary containing the filtered bucket assignments for each metric.
         
        """
        filtered_dict = {}
        filtered_bucket_assgn = {}
        for title, metric in metrics_dict.items():
            if title not in metrics_to_plot:
                continue
            filter_mask = self.remove_outliers(metric)
            filtered_dict[title] = metric[filter_mask]
            filtered_bucket_assgn[title] = bucket_dict['bucket_assignments'][filter_mask]
            # if bucket_dict['bucket_assignments'][filter_mask].size == 0:
            #     raise ValueError(f"No data left after outlier removal for metric {title}.")
            
        print(filtered_dict.keys())
        
        return filtered_dict, filtered_bucket_assgn

    def remove_outliers(self, data, method='std'):
        """
        Removes outliers from the data using standard deviation or IQR.

            Parameters:
                data (np.array): Array of data to filter.
                method (str): Method to use for filtering. Choose 'std' or 'iqr'.
            
            Returns:
                filter_mask (np.array): Boolean array indicating which data points to keep.
        
        """
        if method == 'std':
            mean = np.mean(data)
            std_dev = np.std(data)
            lower_bound = mean - 3 * std_dev
            upper_bound = mean + 3 * std_dev
        elif method == 'iqr':
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:
            raise ValueError("Invalid method. Choose 'std' or 'iqr'.")
        
        filter_mask = (data >= lower_bound) & (data <= upper_bound)
        return filter_mask

    def precompute_bin_edges(self, num_bins=15):
        """
        Computes bin edges for histograms based on the global min and max of the data for each metric.
        Handles the case where we are visualizing two experiments.

            Parameters:
                num_bins (int): Number of bins to use for the histograms.
            
            Returns:
                bin_edges_dict (dict): Dictionary containing the bin edges for each metric.
        """
        bin_edges_dict = {}
        
        combined_metrics = {}

        # Comncatenating second metrics_dict if available
        if self.metrics_dict2 is not None:
            for key in set(self.metrics_dict.keys()).intersection(self.metrics_dict2):
                combined_metrics[key] = np.concatenate((self.metrics_dict[key], self.metrics_dict2[key]))
        else:
            combined_metrics = self.metrics_dict
        
        for title, metric in combined_metrics.items():
            # Determine the global min and max across all windows if applicable
            global_min = np.min(metric)
            global_max = np.max(metric)

            # Define bins explicitly in a range that covers all data for the metric
            bins = np.linspace(global_min, global_max, num=num_bins + 1) 
            bin_edges_dict[title] = bins

        return bin_edges_dict

    def find_max_frequency_per_metric(self, bin_edges_dict):
        """
        Finds the maximum frequency for each metric to set the y-axis limits for plotting. Handles the 
        case where we are visualizing two experiments.

            Parameters:
                bin_edges_dict (dict): Dictionary containing the bin edges for each metric.
            
            Returns:
                max_freq_per_metric (dict): Dictionary containing the maximum frequency for each metric.
        """
        max_freq_per_metric = {}

        for key, edges in bin_edges_dict.items():
            max_freq = 0

            for bucket_index in range(self.num_buckets):
                bucket_indices = np.where(self.bucket_assignments[key] == bucket_index)[0]
                metric_data = self.metrics_dict[key][bucket_indices]
                counts, _ = np.histogram(metric_data, bins=edges)
                max_freq = max(max_freq, np.max(counts))

            if self.metrics_dict2  is not None:
                for bucket_index in range(self.num_buckets):
                    bucket_indices = np.where(self.bucket_assignments2[key] == bucket_index)[0]
                    metric_data = self.metrics_dict2[key][bucket_indices]
                    counts, _ = np.histogram(metric_data, bins=edges)
                    max_freq = max(max_freq, np.max(counts))

            max_freq_per_metric[key] = max_freq

        return max_freq_per_metric

    def plot_bucket_data(self, bucket_index, fig, axes, bin_edges_dict, max_freq_per_metric):
        """
        Plotting method to plot the data for a given bucket. Handles the case where we are visualizing two experiments.

            Parameters:
                bucket_index (int): The index of the bucket to plot.
                fig (plt.Figure): The figure object to plot on.
                axes (list): List of axes objects to plot on.
                bin_edges_dict (dict): Dictionary containing the bin edges for each metric.
                max_freq_per_metric (dict): Dictionary containing the maximum frequency for each metric.
            
        
        """
        for ax, key in zip(axes, self.available_metrics_to_plot):
            ax.clear()

            # Determine the number of datasets and whether previous buckets are being plotted
            num_datasets = 1 + (self.metrics_dict2 is not None and key in self.metrics_dict2)
            num_bucket_groups = 1 + (self.plot_previous_bucket and bucket_index > 0)

            # Calculate the bar width based on the number of groups and available bin width
            bin_widths = np.diff(bin_edges_dict[key])
            min_bin_width = min(bin_widths)
            # Adjust bar width to fit the number of datasets and bucket groups
            bar_width = (min_bin_width / (num_datasets * num_bucket_groups)) * 0.8

            # Calculate the initial offset to start plotting the bars from
            initial_offset = -((bar_width * num_datasets) * num_bucket_groups) / 2 + (bar_width / 2)

            for group_index in range(num_bucket_groups):
                for dataset_index in range(num_datasets):
                    # Determine which dataset and bucket index we are plotting
                    dataset = self.metrics_dict if dataset_index == 0 else self.metrics_dict2
                    bucket_assgn = self.bucket_assignments if dataset_index == 0 else self.bucket_assignments2
                    bucket_id = bucket_index - group_index

                    # Calculate offset for each bar within the group
                    offset = initial_offset + (group_index * (bar_width * num_datasets)) + (dataset_index * bar_width)

                    # Plot the bars if data is available
                    bucket_indices = np.where(bucket_assgn[key] == bucket_id)[0]
                    if len(bucket_indices) > 0:
                        bucket_data = dataset[key][bucket_indices]
                        color = '#307EC7' if dataset_index == 0 else '#dc143c'
                        alpha = 0.75 if group_index == 0 else 0.5  # Current bucket more opaque than previous
                        label_prefix = 'User' if dataset_index == 0 else 'Expert'
                        label_suffix = 'Current' if group_index == 0 else 'Previous'
                        label = f'{label_prefix} {label_suffix}'
                        self._plot_bar(ax, bucket_data, bin_edges_dict[key], bar_width, offset, color, alpha, label)

                    ax.set_xticks(bin_edges_dict[key])
                    ax.set_xlim(min(bin_edges_dict[key]), max(bin_edges_dict[key]))
                    ax.set_xticklabels([f'{tick:.2f}' for tick in bin_edges_dict[key]], rotation=45, ha='right')


            ax.set_title(self.titles[key])
            ax.set_ylim(0, max_freq_per_metric[key])
            ax.set_xlabel(self.xlabels[key])
            ax.set_ylabel('Frequency')
            ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
            ax.legend()

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.suptitle(f'Stroke Metrics for Segment {bucket_index + 1}/{self.num_buckets}', fontsize=16)
        fig.canvas.draw_idle()

    def _plot_bar(self, ax, data, bin_edges, bar_width, offset, color, alpha, label):
        """
        Helper method to plot a bar chart for a given metric.

        Parameters:
            ax (plt.Axes): The axes object to plot on.
            data (np.array): The data to plot.
            bin_edges (np.array): The bin edges for the histogram.
            bar_width (float): The width of the bars.
            offset (float): The offset to apply to the bars. As we are manually plotting the bars.
            color (str): The color of the bars.
            alpha (float): The transparency of the bars.
            label (str): The label for the bars.
        """
        counts, _ = np.histogram(data, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers + offset, counts, width=bar_width, align='center', color=color, alpha=alpha, label=label)

    def interactive_plot_buckets(self, num_bins=15):
        """
        Creates an interactive visualization of the stroke metrics for each bucket, that allows
        one to use a slider to navigate through the buckets. Handles the case where we are visualizing two experiments. Also
        provides the option to plot the previous bucket.

            Parameters:
                num_bins (int): The number of bins to use for the histograms.
        
        
        """
        ncols = np.ceil(len(self.metrics_dict) / 2).astype(int)
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(15, 10))
        # plt.subplots_adjust(bottom=0.2)
        axes = axes.flatten()

        bin_edges_dict = self.precompute_bin_edges(num_bins)
        max_freq_per_metric = self.find_max_frequency_per_metric(bin_edges_dict)

        def update(val):
            bucket_index = int(slider.val)
            self.plot_bucket_data(bucket_index, fig, axes, bin_edges_dict, max_freq_per_metric)

        self.plot_bucket_data(0, fig, axes, bin_edges_dict, max_freq_per_metric)
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Segment', 0, self.num_buckets - 1, valinit=0, valstep =1)
        slider.on_changed(update)
        plt.show()

def rgb_to_hex(r, g, b):
    """
    Helper function to convert RGB values to hex format.
    """
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def plot_3d_vx_rmvd(exp):
    """
    Plots the removed voxels in 3D space for a given experiment
    """
    vrm = exp.v_rm_locs
    vcol = exp.v_rm_colors

    colors = [None for _ in range(vcol.shape[0])]
    for i, c in enumerate(vcol):
        colors[i] = rgb_to_hex(c[1], c[2], c[3])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(vrm[:, 1], vrm[:, 2], vrm[:, 3], alpha=.3, c=colors)
    # ax.scatter([1, 2, 3], [5, 6, 4], [9, 5, 4], label="X")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.legend(["Dura", "Tegmen"])
    plt.title('Removed Voxels')
    plt.show()