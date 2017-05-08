%

addpath('/home/xyy/matcode/stft_mrfft/');

sr1 = 44100;

x1 = sin(1:1e6).';
sz_dat = 2^10;

f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:sz_dat).'-0.5)/sz_dat - 0.5);
wnd = wnd / sum(wnd .* wnd);

[s1, fqs1] = sft_wnd(x1, wnd, [], [], 'audio');
[s1_, fqs1_] = sft_wnd_c(x1, wnd, [], [], 'audio');

%fqs1 = fqs1(1:end/2) * sr1;
%s1   = 10*log10(s1(1:end/2));

%figure(1);
%plot(fqs1, s1);

