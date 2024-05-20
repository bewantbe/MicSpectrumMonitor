% Simple spectrum analysis
% detailed formula for the lower bound of the spectrum
% considering the jitter and quantization noise

% Note: Spectrum computation didn't consider the correction of
#       DC component and the Nyquist frequency.

sr = 250e3;       % sample rate
lw = 65536;       % window length
leap = lw / 2;    % control the overlap ratio
f0 = 40000;        % Hz, the signal freq
imax = 32768;     % 16bit <-> -32768 ~ 32767
ftype = @single;  % @single or @double

w = ftype((1+cos(2*pi*((0:lw-1)-lw/2)/lw))/2);  % Hann window
%w = ftype(ones(1, lw));  % Rectangular window
%w = ftype(acos(cos(2*pi*(0:lw-1)/lw)));  % Bartlett window
normalize_peak = 1;   % 2: sine = 0dB, 1: square = 0dB

T = 10;           % >= lw/sr
s_xxw = zeros(1, lw);

for k = 1 : floor((T*sr-lw)/leap)+1
    tt = (0 : lw-1) / sr + k * leap / sr;
    xx = 1.0 * sin(f0*2*pi * tt);                 % test signal
    xx = xx + (rand(1, lw)-0.5) / imax;           % add jitter
    xx = ftype(round(imax * xx)) / imax;          % quantization

    xxw = abs(fft(xx .* w)).^2 * 2 / sum(w)^2 * normalize_peak;
    s_xxw = s_xxw + xxw;
end
s_xxw = s_xxw / k;

lower_bound_approx = 20 * log10(1 / imax / 2) + 10 * log10(1 / lw)

% for [-0.5, 0.5] LSB uniform distribution jitter
min_jitter_energy = (1/imax)^2 * 1/12 * lw * sum(w.^2) * 2 / sum(w)^2;
lower_bound = 10 * log10(min_jitter_energy) ...
            + 10 * log10(1 / lw)      ...   % energy equipartition
            + 10 * log10(2)           ...   % factor for quantization (approx.)
            + 10 * log10(normalize_peak)    % factor for extra normalization

% Estimation error: mean(10*log10(s_xxw)(end/4:3*end/4)) - lower_bound

% Deterministic quantization noise can have harmonics and essentially non-linear
% an eyeball estimation is given here, see it with jitter off.
lower_bound_peak = lower_bound + 34;
                                            
plot(sr*(0:lw-1)/lw, 10*log10(s_xxw), ...
     [0, sr], [lower_bound_peak, lower_bound_peak], 'g',
     [0, sr], [lower_bound, lower_bound], 'k')
xlim([0, sr/2])
xlabel('Hz')
ylabel('dB')
