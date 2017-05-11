%

p2db = @(x) 10*log10(x);  % power to dB
a2db = @(x) 20*log10(x);  % amplitude to dB

# signal
sr = 48000;
dt = 1/sr;
len = 128;
t = linspace(0, (len-1)*dt, len);
x = sin(2*pi*(t/(3*dt) - 0.3));
figure(1); plot(t, x, '-o')

# RMS
rms = sqrt(sumsq(x)/length(x));
fprintf('RMS     = %.6e (%.6e dB)\n', rms, a2db(rms));

# FFT
%wnd = ones(1, len);
f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:len)-0.5)/len - 0.5);

wnd = wnd * (length(wnd) / sum(wnd));
wnd_factor = 4 / sum(wnd) ^ 2;
f = fft(x .* wnd)(1:floor((len+2)/2));

# spectrum
s = (f .* conj(f)) * wnd_factor;
if mod(len, 2) == 0
    s = [s(1)/2, s(2:end-1), s(end)/2];
else
    s = [s(1)/2, s(2:end)];
end
figure(2); plot((0:length(s)-1)/len, s)

# RMS from FFT
rms_fft = sqrt(2 * sum(s) / wnd_factor / len / len * len / sum(wnd.^2));
fprintf('RMS_FFT = %.6e (%.6e dB)\n', rms_fft, a2db(rms_fft));

fprintf('zero = %e\n', rms - rms_fft);

%{
% Java

wnd = wnd.len / sum(wnd)

scaler = 2.0*2.0 / (data.length * data.length)
data = fft(wnd .* x)
s = (data[i]*data[i] + data[i+1]*data[i+1]) * scaler

=> spectrum

RMS_FFT
 = sqrt(sum(spectrum) * wnd.len / sum(wnd .^ 2))
%}

# vim: set expandtab shiftwidth=4 softtabstop=4:
