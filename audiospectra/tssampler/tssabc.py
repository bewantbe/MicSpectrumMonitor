"""This module defines the :class:`TSSampler` base class, a common interface for
all time series sampler classes.
"""

import logging
import threading
from abc import ABC, abstractmethod
import inspect

logger = logging.getLogger("TSSampler")

sampler_registry = {}

class ThreadSampler(ABC, threading.Thread):
    """Sampler abstract base class."""

    # override in sub-class
    sampler_id = None      # sampler identifier, usually a string.

    @abstractmethod
    def loop_fill_queue(self, que):
        """Fill the queue in loop, other threads can read from this queue.

        Parameters
        ----------
        que : queue.Queue, the queue to fill.
        """
    
    @abstractmethod
    def loop_with_callback(self, fn_callback):
        """Loop to feed data to callback function.

        Parameters
        ----------
        fn_callback : a callable, calls like fn(data).
        """
    
    @classmethod
    def from_config(cls, config):
        # Assume constructor accepts configuration parameters as
        # keyword arguments
        return cls(**config)

    def __eq__(self, other):
        # override in sub-class if need special equality comparison
        try:
            return self.get_config() == other.get_config()
        except AttributeError:
            return False

class SampleReader(ABC):

    sampler_id = None
    device_name = None

    @abstractmethod
    def init(self, sample_rate, chunk_size, stream_callback=None, **kwargs):
        """Initialize the reader."""
        # these are mandatory fields
        self.capability = {
            'sample_format': [],
            'sample_rate': [],
            'n_channels': [],
            'period_size': [],
        }
        return self

    @abstractmethod
    def read(self, n_frames = None):
        """Read at most n_frames frames of audio/ADC values etc.
        Return decoded data (numpy array-like), with shape (n_r_frames, n_ch).
        If no n_frames is given, read according to period_size.
        """

    def close(self):
        """Close the reader."""


# Ref. https://github.com/zarr-developers/numcodecs/blob/main/numcodecs/registry.py

def register_sampler(cls, sampler_id = None):
    """Register a Sampler class."""
    if sampler_id is None:
        sampler_id = cls.sampler_id
    logger.debug("Registering sampler '%s'", sampler_id)
    sampler_registry[sampler_id] = cls

def get_sampler(sampler_id):
    """Obtain a sampler for the given configuration.
    """
    if isinstance(sampler_id, str):
        return sampler_registry.get(sampler_id)()
    else:
        # assume sampler_id is a dict
        conf = dict(sampler_id)
        sampler_id = conf.pop('sampler_id', None)
        cls = sampler_registry.get(sampler_id)()
        if cls:
            return cls.init(**conf)
        raise ValueError('sampler not available: %r' % sampler_id)

def get_available_samplers():
    return list(sampler_registry.keys())

def get_all_device_capablity(test):
    # collect basic info
    ts_sampler_dict = {}
    for ts_id, cls_tss in sampler_registry.items():
        ts_sampler_dict[ts_id] = {
            'device_name': cls_tss.device_name,
            'capability': None,
            'default_conf': None
        }
        if not test:
            continue
        # test each device
        try:
            tss = cls_tss()
            cap = tss.capability
            
            signature = inspect.signature(tss.init)
            configurable_keys = [param.name for param in signature.parameters.values()]
            conf = {k:v[0] for k, v in cap.items() if k in configurable_keys}
            logger.info(f'Initializing device "{ts_id}" with conf:\n    {conf}')
            tss.init(**conf)
        except Exception as e:
            logger.info(f'Failed to initialize device "{ts_id}".')
            logger.info(e)
        else:
            ts_sampler_dict[ts_id]['capability'] = tss.capability
            ts_sampler_dict[ts_id]['default_conf'] = \
                {k:v[0] for k, v in tss.capability.items()}
            ts_sampler_dict[ts_id]['configurable_keys'] = configurable_keys
            logger.info(f'Initialized device "{ts_id}", now closing it...')
            tss.close()

    if test:
        ts_sampler_dict = {k: v for k, v in ts_sampler_dict.items() if v['capability']}

    return ts_sampler_dict