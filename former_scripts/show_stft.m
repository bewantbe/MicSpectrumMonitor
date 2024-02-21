%

function [xf, s_t_stft, s_f] = show_stft(fname, sr)

%fid = fopen(fname, 'r');
%x = fread(fid, Inf, '*int16');
%fclose(fid);
%x = double(x) / 32768.0;

if ischar(fname)
  [x sr] = wavread(fname);
  x = x(:, 1);
else
  x = fname;
end

k = 10;
sz_dat = 2^k;
sz_fft = sz_dat;

addpath('/home/xyy/matcode/stft_mrfft/');
addpath('/home/xyy/code/android/audio-analyzer-for-android/util');

f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:sz_dat).'-0.5)/sz_dat - 0.5);
wnd = wnd / sum(wnd .* wnd);

[xf, s_t_stft, s_f] = stft_wnd_c(x, wnd, sz_fft/4);

stft_db = 20 * log10(abs(xf(1:end/2, :)));

s_t_stft_show = s_t_stft / sr;
s_f_show = s_f(1:sz_fft/2) * sr;

imagesc(s_t_stft_show, s_f_show, stft_db);
set(gca,'YDir','normal')
colormap(inferno());
colorbar();
caxis([-130, 0]);
%axis([min(s_t), max(s_t) 0 1/dt/2]);

% vim: set expandtab shiftwidth=2 softtabstop=2:
