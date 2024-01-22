"""This module defines the :class:`TSSampler` base class, a common interface for
all time series sampler classes.
"""

import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger("Sampler")

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

    @abstractmethod
    def init(self, sample_rate, chunk_size, stream_callback=None, **kwargs):
        """Initialize the reader."""

    @abstractmethod
    def read(self, n_frames):
        """Read at most n_frames frames of audio/ADC values etc.
        Return decoded data.
        """
    
    @abstractmethod
    def close(self):
        """Close the reader."""


# Ref. https://github.com/zarr-developers/numcodecs/blob/main/numcodecs/registry.py

def register_sampler(cls, sampler_id = None):
    """Register a Sampler class."""
    if sampler_id is None:
        sampler_id = cls.sampler_id
    logger.debug("Registering sampler '%s'", sampler_id)
    sampler_registry[sampler_id] = cls

def get_sampler(config):
    """Obtain a sampler for the given configuration.

    Parameters: config: dick-like, configuration object.

    Return: the sampler
    """
    config = dict(config)
    sampler_id = config.pop('id', None)
    cls = sampler_registry.get(sampler_id)
    if cls:
        return cls.from_config(config)
    raise ValueError('sampler not available: %r' % sampler_id)
