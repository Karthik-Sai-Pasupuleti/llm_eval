import numpy as np
from scipy.signal import butter, filtfilt
from typing import Union, Tuple, List
import pandas as pd


def calculate_blink_stats_all(ear_values: np.ndarray, ear_threshold: float) -> tuple:
    """
    Calculate mean, standard deviation, and maximum blink durations (in seconds)
    from EAR values over a 60-second window.

    Args:
        ear_values (np.ndarray): Array of EAR values.
        ear_threshold (float): EAR threshold below which eyes are considered closed.

    Returns:
        tuple: (mean_blink_duration, std_blink_duration, max_blink_duration)
    """
    if len(ear_values) == 0:
        return 0.0, 0.0, 0.0

    total_frames = np.sum([ear is not None for ear in ear_values])
    if total_frames == 0:
        return 0.0, 0.0, 0.0

    fps = total_frames / 60.0
    blink_durations = []
    consec_count = 0

    for ear in ear_values:
        if ear is not None:
            if ear < ear_threshold:
                consec_count += 1
            else:
                if consec_count > 0:
                    blink_durations.append(consec_count / fps)
                    consec_count = 0
        else:
            if consec_count > 0:
                blink_durations.append(consec_count / fps)
                consec_count = 0

    if consec_count > 0:
        blink_durations.append(consec_count / fps)

    if not blink_durations:
        return 0.0, 0.0, 0.0

    mean_duration = float(np.mean(blink_durations))
    std_duration = float(np.std(blink_durations))
    max_duration = float(np.max(blink_durations))
    return mean_duration, std_duration, max_duration


def calculate_perclos(
    ear_values: np.ndarray, ear_threshold: float, min_consec_frames: int
) -> float:
    """
    Calculate PERCLOS (Percentage of Eye Closure) from stored EAR values.

    Args:
        ear_values (np.ndarray): Array of EAR values sampled over time.
        ear_threshold (float): EAR threshold below which eyes are considered closed.
        min_consec_frames (int): Minimum consecutive frames below threshold to count as closed.

    Returns:
        float: PERCLOS percentage over the EAR values window.
    """
    closed_frames = 0
    consec_count = 0

    for ear in ear_values:
        if ear is not None:  # frames with no mediapipe mesh records aa none
            if ear < ear_threshold:  # eye closed
                consec_count += 1
            else:
                if consec_count >= min_consec_frames:  # eye opened
                    closed_frames += (
                        consec_count  # add the consecutive frames to closed frames
                    )
                consec_count = 0  # restart the count

    # Account for closing at the end of the window
    if consec_count >= min_consec_frames:
        closed_frames += consec_count

    total_frames = np.sum([ear is not None for ear in ear_values])

    if total_frames == 0:
        return 0.0

    perclos = (closed_frames / total_frames) * 100.0
    return perclos






def approx_entropy(time_series: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Fast Approximate Entropy (ApEn) computation.
    Based on Pincus (1991) and same logic as your code, but vectorized.

    Parameters
    ----------
    time_series : np.ndarray
        Input steering angle (or other) time series.
    m : int, optional
        Embedding dimension (run length). Default = 2.
    r : float, optional
        Tolerance threshold. Default = 0.2 * std(time_series).

    Returns
    -------
    float
        Approximate entropy value (0 → regular, 1 → irregular)
    """
    x = np.asarray(time_series, dtype=float)
    n = len(x)

    if r is None:
        r = 0.2 * np.std(x)

    # Build sequences of length m and m+1
    def _embed(x, dim):
        return np.array([x[i : i + dim] for i in range(n - dim + 1)])

    xm = _embed(x, m)
    xm1 = _embed(x, m + 1)

    # Compute Chebyshev (max) distance efficiently
    def _phi(X):
        # Broadcasting pairwise distances using numpy
        dists = np.max(np.abs(X[:, None, :] - X[None, :, :]), axis=2)
        C = np.sum(dists <= r, axis=0) / (len(X))
        # Avoid log(0)
        C = np.where(C == 0, 1e-10, C)
        return np.mean(np.log(C))

    phi_m = _phi(xm)
    phi_m1 = _phi(xm1)

    return abs(phi_m1 - phi_m)



def steering_reversal_rate(
    steering_angle: np.ndarray,
    gap_size: float = 3.0,
    lowpass_cutoff: float = 0.6,
    filter_order: int = 2,
) -> float:
    """
    Simple and clear implementation of the Markkula & Engström (2006)
    Steering Wheel Reversal Rate (SRR) algorithm.

    Parameters
    ----------
    steering_angle : np.ndarray
        Steering wheel angle signal (degrees)
    gap_size : float, optional
        Minimum angular change to count as a reversal (default 3.0° for visual load)
    lowpass_cutoff : float, optional
        Low-pass filter cutoff frequency (Hz). Use 0.6 Hz for visual, 2 Hz for cognitive.
    filter_order : int, optional
        Butterworth filter order (default 2)

    Returns
    -------
    float
        Steering wheel reversal rate (reversals per minute)
    """

    # === 1. LOW-PASS FILTER ===
    fs = len(steering_angle) / 60.0  # Sampling frequency (Hz), assuming 60s window
    nyquist = 0.5 * fs
    normal_cutoff = lowpass_cutoff / nyquist
    b, a = butter(filter_order, normal_cutoff, btype="low", analog=False)
    filtered = filtfilt(b, a, steering_angle)

    # === 2. FIND STATIONARY POINTS (local minima & maxima) ===
    diff = np.diff(filtered)
    sign_diff = np.sign(diff)

    stationary_points = []
    for i in range(1, len(sign_diff)):
        if sign_diff[i] != sign_diff[i - 1]:  # sign change → extremum
            stationary_points.append(i)

    if len(stationary_points) < 2:
        return 0.0  # no reversals possible

    # === 3. COUNT REVERSALS (for both up and down directions) ===
    def count_reversals(theta_values, gap):
        count = 0
        k = 0
        for l in range(1, len(theta_values)):
            if theta_values[l] - theta_values[k] >= gap:
                count += 1
                k = l
            elif theta_values[l] < theta_values[k]:
                k = l
        return count

    # Extract steering values at stationary points
    theta_vals = filtered[stationary_points]

    # Count upward and downward reversals
    n_up = count_reversals(theta_vals, gap_size)
    n_down = count_reversals(-theta_vals, gap_size)

    total_reversals = n_up + n_down

    # === 4. CALCULATE REVERSALS PER MINUTE ===
    duration_min = len(filtered) / (fs * 60.0)
    reversal_rate = total_reversals / duration_min if duration_min > 0 else 0.0

    return reversal_rate



# create a new column label by checking karthik_drowsiness_level and Vanchha_drowsiness_level if they are same then keep that value otherwise remove that row
def checking_conflict(row: pd.Series) -> Union[str, np.nan]:
    """checking both annotators labelled same label or different.

    Args:
        row (pd.Series): data

    Returns:
        Union[str, np.nan]: if both annotators labelled same label and no nan values returns that label otherwise return nan
    """
    if (row['karthik_drowsiness_level'] == row['Vanchha_drowsiness_level']) and (not pd.isna(row['karthik_drowsiness_level']) and not pd.isna(row['Vanchha_drowsiness_level'])):
        return row['karthik_drowsiness_level']
    else:
        return np.nan