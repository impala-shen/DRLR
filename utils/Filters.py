import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque


class RealTimeFilter:
    """Real-time filter that processes one sample at a time"""

    def __init__(self, sample_rate=1000):
        self.fs = sample_rate
        self.reset()

    def reset(self):
        """Reset all filter states"""
        # RC Filter state
        self.rc_y_prev = 0.0

        # EMA Filter state
        self.ema_y_prev = 0.0

        # Moving Average states
        self.sma_buffer = deque()
        self.sma_sum = 0.0
        self.sma_window_size = 10  # default

        # Weighted MA state
        self.wma_buffer = deque()
        self.wma_weights = [1, 2, 3, 4, 5]  # default weights

        # Adaptive MA states
        self.adaptive_buffer = deque()
        self.adaptive_window = 5  # current window size
        self.adaptive_min_window = 5
        self.adaptive_max_window = 20
        self.adaptive_threshold = 0.1

        # Butterworth filter states
        self.butter_x_buffer = deque()
        self.butter_y_buffer = deque()
        self.butter_b_coeffs = None
        self.butter_a_coeffs = None
        self.butter_initialized = False


class RCFilter(RealTimeFilter):
    """Real-time RC Low-Pass Filter"""

    def __init__(self, sample_rate=1000, cutoff_freq=50):
        super().__init__(sample_rate)
        self.cutoff_freq = cutoff_freq
        self._calculate_alpha()

    def _calculate_alpha(self):
        """Calculate RC filter coefficient"""
        dt = 1.0 / self.fs
        rc = 1.0 / (2 * np.pi * self.cutoff_freq)
        self.alpha = dt / (rc + dt)

    def set_cutoff(self, cutoff_freq):
        """Change cutoff frequency"""
        self.cutoff_freq = cutoff_freq
        self._calculate_alpha()

    def process_sample(self, x):
        """Process single input sample"""
        # y[n] = α*x[n] + (1-α)*y[n-1]
        y = self.alpha * x + (1 - self.alpha) * self.rc_y_prev
        self.rc_y_prev = y
        return y


class EMAFilter(RealTimeFilter):
    """Real-time Exponential Moving Average Filter"""

    def __init__(self, sample_rate=1000, alpha=0.1):
        super().__init__(sample_rate)
        self.alpha = alpha

    def set_alpha(self, alpha):
        """Change smoothing factor (0 < alpha < 1)"""
        self.alpha = alpha

    def process_sample(self, x):
        """Process single input sample"""
        # y[n] = α*x[n] + (1-α)*y[n-1]
        y = self.alpha * x + (1 - self.alpha) * self.ema_y_prev
        self.ema_y_prev = y
        return y


class SMAFilter(RealTimeFilter):
    """Real-time Simple Moving Average Filter"""

    def __init__(self, sample_rate=1000, window_size=10):
        super().__init__(sample_rate)
        self.window_size = window_size

    def set_window_size(self, window_size):
        """Change window size"""
        self.window_size = window_size
        # Adjust buffer if needed
        while len(self.sma_buffer) > window_size:
            removed = self.sma_buffer.popleft()
            self.sma_sum -= removed

    def process_sample(self, x):
        """Process single input sample"""
        # Add new sample
        self.sma_buffer.append(x)
        self.sma_sum += x

        # Remove old sample if buffer is full
        if len(self.sma_buffer) > self.window_size:
            removed = self.sma_buffer.popleft()
            self.sma_sum -= removed

        # Return average
        return self.sma_sum / len(self.sma_buffer)


class WMAFilter(RealTimeFilter):
    """Real-time Weighted Moving Average Filter"""

    def __init__(self, sample_rate=1000, weights=None):
        super().__init__(sample_rate)
        if weights is None:
            weights = [1, 2, 3, 4, 5]  # Default: more weight to recent samples
        self.weights = weights
        self.window_size = len(weights)

    def set_weights(self, weights):
        """Change weights (most recent sample gets weights[0])"""
        self.weights = weights
        self.window_size = len(weights)
        # Adjust buffer if needed
        while len(self.wma_buffer) > self.window_size:
            self.wma_buffer.popleft()

    def process_sample(self, x):
        """Process single input sample"""
        # Add new sample
        self.wma_buffer.append(x)

        # Remove old sample if buffer is full
        if len(self.wma_buffer) > self.window_size:
            self.wma_buffer.popleft()

        # Calculate weighted average
        if len(self.wma_buffer) == 0:
            return 0.0

        # Apply weights (most recent sample gets first weight)
        buffer_list = list(self.wma_buffer)
        weighted_sum = 0.0
        weight_sum = 0.0

        for i, sample in enumerate(reversed(buffer_list)):
            if i < len(self.weights):
                weight = self.weights[i]
                weighted_sum += weight * sample
                weight_sum += weight

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0


class AdaptiveMaFilter(RealTimeFilter):
    """Real-time Adaptive Moving Average Filter"""

    def __init__(self, sample_rate=1000, min_window=5, max_window=20, threshold=0.1):
        super().__init__(sample_rate)
        self.min_window = min_window
        self.max_window = max_window
        self.threshold = threshold
        self.current_window = min_window

    def process_sample(self, x):
        """Process single input sample with adaptive window"""
        # Add new sample
        self.adaptive_buffer.append(x)

        # Keep buffer size manageable
        if len(self.adaptive_buffer) > self.max_window * 2:
            # Remove oldest samples beyond what we need
            while len(self.adaptive_buffer) > self.max_window:
                self.adaptive_buffer.popleft()

        # Calculate variance of recent samples to adapt window size
        if len(self.adaptive_buffer) >= self.min_window:
            recent_samples = list(self.adaptive_buffer)[-self.min_window:]
            variance = np.var(recent_samples)

            # Adapt window size based on variance
            if variance > self.threshold:
                # High variance -> reduce window for responsiveness
                self.current_window = max(self.min_window, self.current_window - 1)
            else:
                # Low variance -> increase window for smoothing
                self.current_window = min(self.max_window, self.current_window + 1)

        # Calculate moving average with current window size
        buffer_list = list(self.adaptive_buffer)
        window_data = buffer_list[-self.current_window:]
        return sum(window_data) / len(window_data)


class ButterworthFilter(RealTimeFilter):
    """Real-time Butterworth Filter"""

    def __init__(self, sample_rate=1000, cutoff_freq=50, order=2):
        super().__init__(sample_rate)
        self.cutoff_freq = cutoff_freq
        self.order = order
        self._design_filter()

    def _design_filter(self):
        """Design Butterworth filter coefficients"""
        nyquist = 0.5 * self.fs
        normalized_cutoff = self.cutoff_freq / nyquist
        self.butter_b_coeffs, self.butter_a_coeffs = signal.butter(
            self.order, normalized_cutoff, btype='low')

        # Initialize buffers with correct sizes
        self.butter_x_buffer = deque([0.0] * len(self.butter_b_coeffs),
                                     maxlen=len(self.butter_b_coeffs))
        self.butter_y_buffer = deque([0.0] * (len(self.butter_a_coeffs) - 1),
                                     maxlen=len(self.butter_a_coeffs) - 1)
        self.butter_initialized = True

    def set_parameters(self, cutoff_freq=None, order=None):
        """Change filter parameters"""
        if cutoff_freq is not None:
            self.cutoff_freq = cutoff_freq
        if order is not None:
            self.order = order
        self._design_filter()

    def process_sample(self, x):
        """Process single input sample"""
        if not self.butter_initialized:
            self._design_filter()

        # Add new input sample (most recent at index 0)
        self.butter_x_buffer.appendleft(x)

        # Calculate output using difference equation
        # y[n] = sum(b[i] * x[n-i]) - sum(a[i] * y[n-i])
        y = 0.0

        # Feed-forward part (numerator)
        for i, b_coeff in enumerate(self.butter_b_coeffs):
            if i < len(self.butter_x_buffer):
                y += b_coeff * self.butter_x_buffer[i]

        # Feedback part (denominator, excluding a[0])
        for i, a_coeff in enumerate(self.butter_a_coeffs[1:]):
            if i < len(self.butter_y_buffer):
                y -= a_coeff * self.butter_y_buffer[i]

        # Add new output sample (most recent at index 0)
        self.butter_y_buffer.appendleft(y)

        return y
