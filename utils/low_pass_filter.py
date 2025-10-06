import numpy as np


class LowPassFilter:
    def __init__(self, cutoff_freq: float, sample_rate: float):
        """
        Initialize low-pass filter

        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sampling rate in Hz
        """
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate

        # Compute filter coefficient
        nyquist = 0.5 * sample_rate
        normalized_freq = cutoff_freq / nyquist
        self.alpha = normalized_freq / (1 + normalized_freq)

        # Previous filtered value
        self.prev_filtered_value = 0

    def filter(self, new_value: float) -> float:
        """
        Apply single-pole low-pass filter

        Args:
            new_value: New input signal value

        Returns:
            Filtered signal value
        """
        filtered_value = (self.alpha * new_value +
                          (1 - self.alpha) * self.prev_filtered_value)

        self.prev_filtered_value = filtered_value
        return filtered_value

    def apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply filter to entire signal array

        Args:
            signal: Input signal array

        Returns:
            Filtered signal array
        """
        filtered_signal = np.zeros_like(signal)
        self.prev_filtered_value = signal[0]

        for i in range(len(signal)):
            filtered_signal[i] = self.filter(signal[i])

        return filtered_signal


def plot_filter_comparison(original_signal, filtered_signal):
    """
    Visualize original and filtered signals
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(original_signal, label='Original Signal')
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.title('Low-Pass Filter Comparison')
    plt.legend()
    plt.show()


# # Example usage
# if __name__ == "__main__":
#     # Generate sample noisy signal
#     np.random.seed(42)
#     time = np.linspace(0, 10, 1000)
#     original_signal = np.sin(2 * np.pi * 5 * time) + np.random.normal(0, 0.5, time.shape)
#
#     # Create low-pass filter
#     lpf = LowPassFilter(cutoff_freq=10, sample_rate=100)
#     filtered_signal = lpf.apply_filter(original_signal)
#
#     # Plot results
#     plot_filter_comparison(original_signal, filtered_signal)