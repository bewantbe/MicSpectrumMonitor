%

function [x sr] = show_stft(fname)

%fid = fopen(fname, 'r');
%x = fread(fid, Inf, '*int16');
%fclose(fid);
%x = double(x) / 32768.0;

[x sr] = wavread(fname);
x = x(:, 1);

k = 10;
sz_dat = 2^k;
sz_fft = sz_dat;

addpath('/home/xyy/matcode/stft_mrfft/');
addpath('/home/xyy/code/android/audio-analyzer-for-android/util');

f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:sz_dat).'-0.5)/sz_dat - 0.5);
wnd = wnd / sum(wnd .* wnd);

[xf, s_t_stft, s_f] = stft_wnd_c(x, wnd);
xf = xf.';

stft_db = 20 * log10(abs(xf(:, 1:end/2)'));

dt = sz_dat/sr/2;
s_t_stft = s_t_stft * dt;
s_f = 1/dt * s_f(1:sz_fft/2);

imagesc(s_t_stft, s_f, stft_db);
set(gca,'YDir','normal')
colormap(inferno());
colorbar();
caxis([-130, 0]);
%axis([min(s_t), max(s_t) 0 1/dt/2]);

% vim: set expandtab shiftwidth=2 softtabstop=2:
