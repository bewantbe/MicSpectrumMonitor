# load all known samplers
# Usage 1:
#   from tssamplers import get_sampler
#   sam = get_sampler('mic')
#   sam.init(sample_rate = 48000)
#   vals = sam.read(1024)           # read 1024 frames
# Usage 2:
#   conf = {'sampler_id': 'mic', 'sample_rate': 48000}
#   sam = get_sampler(conf)
#   vals = sam.read(1024)

import logging

from .tssabc import register_sampler, get_sampler, sampler_registry

from .ideal_source import SineSource, WhiteSource
register_sampler(ideal_source.SineSource)
register_sampler(ideal_source.WhiteSource)

# test we are on windows or linux
import platform
if platform.system() == 'Windows':
    from .audio_win import MicReader
    register_sampler(MicReader)
    register_sampler(MicReader, 'mic')
else:
    from .alsa_linux import AlsaAudio
    register_sampler(AlsaAudio)
    register_sampler(AlsaAudio, 'mic')

try:
    from .read_ad7606c import AD7606CReader
except ImportError:
    logging.warning('AD7606C reader is not available')
else:
    register_sampler(AD7606CReader)