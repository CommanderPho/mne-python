"""Reading tools from XDF files."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from pathlib import Path

import numpy as np

try:
    import pyxdf
except ImportError:
    pyxdf = None

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import create_info
from ...annotations import Annotations
from ...utils import _check_fname, _validate_type, fill_doc, logger, verbose, warn
from ..base import BaseRaw


def _check_pyxdf():
    """Check if pyxdf is installed."""
    if pyxdf is None:
        raise ImportError(
            "pyxdf is required to read XDF files. "
            "Install it using: pip install pyxdf"
        )


def _get_stream_info(stream):
    """Extract channel information from XDF stream."""
    info_dict = stream.get("info", {})
    
    # Get channel names
    ch_names = []
    try:
        desc = info_dict.get("desc", [{}])[0]
        channels = desc.get("channels", [{}])[0]
        channel_list = channels.get("channel", [])
        if channel_list:
            for ch in channel_list:
                label = ch.get("label", [""])[0]
                if label:
                    ch_names.append(label)
                else:
                    # Fallback to channel index
                    ch_names.append(f"CH{len(ch_names) + 1}")
        else:
            # No channel info, create default names
            n_channels = stream.get("time_series", np.array([])).shape[1] if len(stream.get("time_series", [])) > 0 else 0
            ch_names = [f"CH{i+1}" for i in range(n_channels)]
    except (KeyError, IndexError, TypeError):
        # Fallback: create default channel names
        n_channels = stream.get("time_series", np.array([])).shape[1] if len(stream.get("time_series", [])) > 0 else 0
        ch_names = [f"CH{i+1}" for i in range(n_channels)]
    
    # Get sampling rate
    try:
        sfreq = float(info_dict.get("nominal_srate", [0.0])[0])
        if sfreq == 0.0:
            # Try to estimate from timestamps
            time_stamps = stream.get("time_stamps", [])
            if len(time_stamps) > 1:
                sfreq = 1.0 / np.mean(np.diff(time_stamps))
            else:
                sfreq = 1000.0  # Default fallback
                warn("Could not determine sampling rate, using 1000 Hz")
    except (KeyError, IndexError, ValueError, TypeError):
        sfreq = 1000.0
        warn("Could not determine sampling rate, using 1000 Hz")
    
    # Get channel types (default to EEG)
    ch_types = ["eeg"] * len(ch_names)
    
    # Try to infer channel types from names
    for i, name in enumerate(ch_names):
        name_lower = name.lower()
        if any(x in name_lower for x in ["eog", "eye"]):
            ch_types[i] = "eog"
        elif any(x in name_lower for x in ["ecg", "ekg"]):
            ch_types[i] = "ecg"
        elif any(x in name_lower for x in ["emg"]):
            ch_types[i] = "emg"
        elif any(x in name_lower for x in ["stim", "trigger", "marker"]):
            ch_types[i] = "stim"
        elif any(x in name_lower for x in ["misc", "aux"]):
            ch_types[i] = "misc"
    
    return ch_names, ch_types, sfreq


@fill_doc
class RawXDF(BaseRaw):
    """Raw object from XDF file.

    Parameters
    ----------
    input_fname : path-like
        Path to the XDF file.
    stream_id : int | None
        ID of the stream to load. If None, the first data stream is used.
        Use ``streams`` parameter to see available streams.
    %(preload)s
    %(verbose)s

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods.
    mne.io.read_raw_xdf : Recommended way to read XDF files.

    Notes
    -----
    XDF files can contain multiple streams with different data types and
    sampling rates. Use the ``stream_id`` parameter to select a specific
    stream, or check available streams using the ``streams`` parameter
    in :func:`mne.io.read_raw_xdf`.
    """

    @verbose
    def __init__(self, input_fname, stream_id=None, preload=False, *, verbose=None):
        _check_pyxdf()
        input_fname = str(_check_fname(input_fname, "read", True, "input_fname"))
        _validate_type(stream_id, (int, type(None)), "stream_id")
        
        logger.info(f"Reading XDF file: {input_fname}")
        
        # Load XDF file
        streams, header = pyxdf.load_xdf(input_fname)
        
        if len(streams) == 0:
            raise ValueError(f"No streams found in XDF file: {input_fname}")
        
        # Find data streams (exclude marker streams)
        data_streams = []
        for i, stream in enumerate(streams):
            stream_type = stream.get("info", {}).get("type", [""])[0].lower()
            if stream_type not in ["markers", "marker"]:
                data_streams.append((i, stream))
        
        if len(data_streams) == 0:
            raise ValueError(
                f"No data streams found in XDF file: {input_fname}. "
                "Only marker streams were found."
            )
        
        # Select stream
        if stream_id is None:
            # Use first data stream
            selected_idx, selected_stream = data_streams[0]
            logger.info(f"Using stream {selected_idx} (first data stream)")
        else:
            # Find stream by ID
            found = False
            for idx, stream in data_streams:
                stream_info = stream.get("info", {})
                stream_id_str = str(stream_id)
                if stream_info.get("stream_id", [""])[0] == stream_id_str:
                    selected_idx, selected_stream = idx, stream
                    found = True
                    break
            if not found:
                available_ids = [
                    s.get("info", {}).get("stream_id", [f"stream_{i}"])[0]
                    for i, s in data_streams
                ]
                raise ValueError(
                    f"Stream ID {stream_id} not found. "
                    f"Available stream IDs: {available_ids}"
                )
            logger.info(f"Using stream {selected_idx} (ID: {stream_id})")
        
        # Extract data
        time_series = selected_stream.get("time_series", [])
        time_stamps = selected_stream.get("time_stamps", [])
        
        if len(time_series) == 0:
            raise ValueError(f"Stream {selected_idx} contains no data")
        
        # Get channel info first to determine expected number of channels
        ch_names, ch_types, sfreq = _get_stream_info(selected_stream)
        
        # Convert to numpy array
        data = np.array(time_series)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        
        # XDF time_series is typically (n_samples, n_channels)
        # MNE expects (n_channels, n_times), so transpose if needed
        # Check which dimension matches the expected number of channels
        n_channels_expected = len(ch_names) if len(ch_names) > 0 else data.shape[1]
        if data.shape[1] == n_channels_expected:
            # Likely (n_samples, n_channels) format, transpose to (n_channels, n_samples)
            data = data.T
        elif data.shape[0] == n_channels_expected:
            # Already in (n_channels, n_samples) format
            pass
        else:
            # Try to infer: if one dimension is much larger, it's likely time
            if data.shape[0] > data.shape[1]:
                data = data.T
        
        n_channels, n_times = data.shape
        
        # Ensure we have the right number of channel names
        if len(ch_names) != n_channels:
            if len(ch_names) > n_channels:
                ch_names = ch_names[:n_channels]
                ch_types = ch_types[:n_channels]
            else:
                # Add default names for missing channels
                for i in range(len(ch_names), n_channels):
                    ch_names.append(f"CH{i+1}")
                    ch_types.append("eeg")
        
        # Create info
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Handle measurement date if available
        if len(time_stamps) > 0:
            # XDF timestamps are relative to file start, not absolute time
            # We can't easily determine the absolute measurement date
            pass
        
        # Store stream info for potential marker extraction
        raw_extras = [{"streams": streams, "selected_stream_idx": selected_idx}]
        
        # Initialize BaseRaw
        super().__init__(
            info,
            preload,
            first_samps=(0,),
            last_samps=[n_times - 1],
            filenames=[input_fname],
            raw_extras=raw_extras,
            orig_format="double",
            verbose=verbose,
        )
        
        # Preload data if requested
        if preload:
            self._data = data.astype(np.float64)
        
        # Extract markers/annotations from marker streams
        self._extract_markers(streams, time_stamps)
    
    def _extract_markers(self, streams, data_time_stamps):
        """Extract markers from marker streams and add as annotations."""
        if len(data_time_stamps) == 0:
            return
        
        # Find marker streams
        marker_onsets = []
        marker_durations = []
        marker_descriptions = []
        
        for stream in streams:
            stream_type = stream.get("info", {}).get("type", [""])[0].lower()
            if stream_type in ["markers", "marker"]:
                marker_times = stream.get("time_stamps", [])
                marker_data = stream.get("time_series", [])
                
                # Convert marker times to sample indices relative to data stream
                data_start_time = data_time_stamps[0]
                data_sfreq = self.info["sfreq"]
                
                for marker_time, marker_info in zip(marker_times, marker_data):
                    # Calculate onset relative to data start
                    relative_time = marker_time - data_start_time
                    if relative_time < 0:
                        continue  # Marker before data start
                    
                    marker_onsets.append(relative_time)
                    marker_durations.append(0.0)
                    
                    # Get marker description
                    if isinstance(marker_info, (list, tuple, np.ndarray)):
                        if len(marker_info) > 0:
                            desc = str(marker_info[0])
                        else:
                            desc = "Marker"
                    else:
                        desc = str(marker_info)
                    marker_descriptions.append(desc)
        
        # Create annotations if we found markers
        if len(marker_onsets) > 0:
            annotations = Annotations(
                onset=marker_onsets,
                duration=marker_durations,
                description=marker_descriptions,
            )
            self.set_annotations(annotations)
            logger.info(f"Added {len(marker_onsets)} markers as annotations")
    
    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of raw data."""
        raw_extra = self._raw_extras[fi]
        streams = raw_extra["streams"]
        selected_idx = raw_extra["selected_stream_idx"]
        
        selected_stream = streams[selected_idx]
        time_series = selected_stream.get("time_series", [])
        
        if len(time_series) == 0:
            raise ValueError(f"Stream {selected_idx} contains no data")
        
        # Convert to numpy array
        stream_data = np.array(time_series)
        if stream_data.ndim == 1:
            stream_data = stream_data[:, np.newaxis]
        
        # XDF time_series is typically (n_samples, n_channels)
        # MNE expects (n_channels, n_samples), so transpose if needed
        n_channels_expected = len(self.ch_names)
        if stream_data.shape[1] == n_channels_expected:
            # Already in (n_samples, n_channels) format, transpose
            stream_data = stream_data.T
        elif stream_data.shape[0] != n_channels_expected:
            # Try transposing if dimensions don't match
            if stream_data.shape[1] == n_channels_expected:
                stream_data = stream_data.T
            else:
                raise ValueError(
                    f"Data shape {stream_data.shape} does not match expected "
                    f"number of channels {n_channels_expected}"
                )
        
        # Extract the requested segment
        block = stream_data[:, start:stop].astype(np.float64)
        
        # Apply calibration
        from ..._fiff.utils import _mult_cal_one
        
        data_view = data[:, :]
        _mult_cal_one(data_view, block, idx, cals, mult)


@fill_doc
def read_raw_xdf(input_fname, stream_id=None, preload=False, *, verbose=None) -> "RawXDF":
    """Read an XDF file.

    Parameters
    ----------
    input_fname : path-like
        Path to the XDF file.
    stream_id : int | None
        ID of the stream to load. If None, the first data stream is used.
        To see available streams, you can load the file with pyxdf directly:
        
        .. code-block:: python
        
            import pyxdf
            streams, header = pyxdf.load_xdf('file.xdf')
            for i, stream in enumerate(streams):
                print(f"Stream {i}: {stream['info'].get('name', ['Unknown'])[0]}")
    %(preload)s
    %(verbose)s

    Returns
    -------
    raw : instance of RawXDF
        A Raw object containing XDF data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawXDF.

    Notes
    -----
    XDF (eXtensible Data Format) is a container format for storing
    multimodal time series data. XDF files can contain multiple streams
    with different data types, sampling rates, and channel configurations.

    If the XDF file contains marker streams, they will be automatically
    converted to MNE annotations.

    Examples
    --------
    >>> import mne
    >>> raw = mne.io.read_raw_xdf('example.xdf', preload=True)
    >>> raw.plot()
    """
    return RawXDF(
        input_fname=input_fname,
        stream_id=stream_id,
        preload=preload,
        verbose=verbose,
    )

