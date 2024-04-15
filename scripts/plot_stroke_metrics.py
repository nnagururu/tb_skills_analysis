import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class StrokeMetricsVisualizer:
    def __init__(self, metrics_dict, bucket_dict, metrics_dict2=None, bucket_dict2=None, plot_previous_bucket=False, 
                 metrics_to_plot = ['length', 'velocity', 'acceleration', 'vxls_removed',
                                    'curvature', 'force', 'angle_wrt_bone', 'angle_wrt_camera']):
        
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
        filtered_dict = {}
        filtered_bucket_assgn = {}
        for title, metric in metrics_dict.items():
            if title not in metrics_to_plot:
                continue
            filter_mask = self.remove_outliers(metric)
            filtered_dict[title] = metric[filter_mask]
            filtered_bucket_assgn[title] = bucket_dict['bucket_assignments'][filter_mask]
            if bucket_dict['bucket_assignments'][filter_mask].size == 0:
                raise ValueError(f"No data left after outlier removal for metric {title}.")
            
        print(filtered_dict.keys())
        
        return filtered_dict, filtered_bucket_assgn

    def remove_outliers(self, data, method='std'):
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
        bin_edges_dict = {}
        
        combined_metrics = {}
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
        counts, _ = np.histogram(data, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers + offset, counts, width=bar_width, align='center', color=color, alpha=alpha, label=label)

    def interactive_plot_buckets(self, num_bins=15):
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

class stroke_metrics_visualizer:
    def __init__(self, metrics_dict, bucket_dict):
        self.metrics_dict = metrics_dict
        self.bucket_assignments = bucket_dict['bucket_assignments']
        self.bucket_ranges = bucket_dict['bucket_ranges']

    def remove_outliers(self, data, method = 'std'):
        """Remove outliers using a threshold of three standard deviations from the mean."""
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
        return data[(data >= lower_bound) & (data <= upper_bound)]

    def plot_metrics(self):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
        fig.suptitle('Histograms of Stroke Metrics')

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Data and titles for each subplot
        metrics = self.metrics_dict.values()
        titles = self.metrics_dict.keys()

        # Generate histograms for each metric
        for ax, metric, title in zip(axes, metrics, titles):
            filtered_metric = self.remove_outliers(np.array(metric), method = 'std')  # Filter out outliers
            ax.hist(filtered_metric, bins='auto')  # 'auto' lets numpy decide the number of bins
            ax.set_title(title)
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Value')

        # Adjust layout to prevent overlap
        # plt.tight_layout()

        # Adjust top margin to accommodate the main title
        plt.subplots_adjust(top=0.9)

        plt.show()

        # # Wait for a key press to terminate
        # print("Press any key or click on the plot to continue...")
        # plt.waitforbuttonpress()

        # # Close the plot
        # plt.close()

    def precompute_bin_edges(self, num_bins = 15):
        bin_edges_dict = {}
        filtered_metrics = {title: self.remove_outliers(np.array(metric), method='std') for title, metric in self.metrics_dict.items()}
        
        for title, metric in filtered_metrics.items():
            # Determine the global min and max across all windows if applicable
            global_min = np.min(metric)
            global_max = np.max(metric)
            # Define bins explicitly in a range that covers all data for the metric
            bins = np.linspace(global_min, global_max, num=num_bins + 1)  # 20 bins example
            bin_edges_dict[title] = bins
        return bin_edges_dict
    

    def find_max_frequency_per_metric(self, bin_edges_dict):
        max_freq_per_metric = {}
        for title, metric in self.metrics_dict.items():
            max_freq = 0
            for bucket_index in range(max(self.bucket_assignments) + 1):
                bucket_indices = np.where(self.bucket_assignments == bucket_index)[0]
                if bucket_indices.size > 0:
                    metric_data = metric[bucket_indices]
                    filtered_metric = self.remove_outliers(metric_data, method='std')
                    counts, _ = np.histogram(filtered_metric, bins=bin_edges_dict[title])
                    max_freq = max(max_freq, counts.max())
            max_freq_per_metric[title] = max_freq
        return max_freq_per_metric
    
    def plot_bucket_data(self, bucket_index, fig, axes, bin_edges_dict, max_freq_per_metric):        
        for ax, (title, metric) in zip(axes, self.metrics_dict.items()):
            ax.clear()  # Clear current axes

            # Calculate bar widths based on bin widths
            bin_widths = np.diff(bin_edges_dict[title])
            bar_width = bin_widths.min() * 0.4  # Choose a width 40% of the minimum bin width

            # Calculate center positions for the bars
            bin_centers = (bin_edges_dict[title][:-1] + bin_edges_dict[title][1:]) / 2

            # Plot previous bucket in lighter color if available
            if bucket_index > 0:
                prev_indices = np.where(self.bucket_assignments == bucket_index - 1)[0]
                if prev_indices.size > 0:
                    stroke_start_prev = prev_indices[0]
                    stroke_end_prev = prev_indices[-1] + 1  # Include the last stroke in the range
                    metric_data_prev = metric[stroke_start_prev:stroke_end_prev]
                    filtered_metric_prev = self.remove_outliers(np.array(metric_data_prev), method='std')
                    counts_prev, _ = np.histogram(filtered_metric_prev, bins=bin_edges_dict[title])
                    # Position previous bars to the left of the bin center
                    ax.bar(bin_centers - bar_width / 2, counts_prev, width=bar_width, align='center', color="#b0c4de", alpha=0.5, label='Previous Bucket')

            # Plot current bucket
            current_indices = np.where(self.bucket_assignments == bucket_index)[0]
            if current_indices.size > 0:
                stroke_start_current = current_indices[0]
                stroke_end_current = current_indices[-1] + 1  # Include the last stroke in the range
                metric_data_current = metric[stroke_start_current:stroke_end_current]
                filtered_metric_current = self.remove_outliers(np.array(metric_data_current), method='std')
                counts_current, _ = np.histogram(filtered_metric_current, bins=bin_edges_dict[title])
                # Position current bars to the right of the bin center
                ax.bar(bin_centers + bar_width / 2, counts_current, width=bar_width, align='center', color="#307EC7", alpha=0.75, label='Current Bucket')

            ax.set_ylim(0, max_freq_per_metric[title])
            ax.set_title(f"{title}")
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Value')
            ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
            ax.legend()

        fig.suptitle(f'Stroke Metrics for Bucket {bucket_index + 1}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.draw()

    def interactive_plot_buckets(self, num_buckets=10, num_bins=15):
        ncols = np.ceil(len(self.metrics_dict)/2).astype(int)
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(15, 10))
        fig.subplots_adjust(bottom=0.25)
        axes = axes.flatten()

        bin_edges_dict = self.precompute_bin_edges(num_bins=num_bins)
        max_freq_per_metric = self.find_max_frequency_per_metric(bin_edges_dict)

        def update(val):
            bucket_index = int(slider.val)
            self.plot_bucket_data(bucket_index, fig, axes, bin_edges_dict, max_freq_per_metric)

        # Initial plot
        self.plot_bucket_data(0, fig, axes, bin_edges_dict, max_freq_per_metric)

        # Setup slider
        num_steps = num_buckets
        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax=ax_slider, label='Bucket Step', valmin=0, valmax=num_steps-1, valinit=0, valstep=1)
        slider.on_changed(update)
        self.slider = slider

        plt.show()