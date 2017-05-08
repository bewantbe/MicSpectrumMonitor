% cpm spectrum

%{
sleep 2
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 44100 -c 1 --duration=10 a1.wav
sleep 1
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 a2.wav
sleep 1
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 96000 -c 1 --duration=10 a3.wav
%}

%{
pacmd set-source-port 1 analog-input-headset-mic
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=0 b00.wav
pacmd list-source-outputs
pacmd set-source-output-volume 454 9000
sleep 3
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b1.wav
sleep 3
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b3.wav
sleep 3
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b4.wav
sleep 1
arecord -vv --dump-hw-params -D 'hw:CARD=U18dB' -f S24_3LE -r 48000 -c 2 --duration=10 b2.wav
%}

fname1 = 'b4.wav';
fname2 = 'b2.wav';
[x1 sr1] = wavread(fname1);
[x2 sr2] = wavread(fname2);
%x2 = x2(:,1) * (10 ^ (0.7/10));

size(x1)
size(x2)

x1 = x1(round(end/4):round(end*3/4));
x2 = x2(round(end/4):round(end*3/4));

rms1 = 20*log10(sqrt(x1' * x1 * 2 / length(x1)));
rms2 = 20*log10(sqrt(x2' * x2 * 2 / length(x1)));
fprintf('rms1 = %5.1f dB\n', rms1);
fprintf('rms2 = %5.1f dB\n', rms2);

addpath('/home/xyy/matcode/stft_mrfft/');

sz_dat = 2^10;

f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:sz_dat).'-0.5)/sz_dat - 0.5);
wnd = wnd / sum(wnd .* wnd);

[s1, fqs1] = sft_wnd(x1, wnd);
fqs1 = fqs1(1:end/2) * sr1;
s1   = 10*log10(s1(1:end/2));
[s2, fqs2] = sft_wnd(x2, wnd);
fqs2 = fqs2(1:end/2) * sr2;
s2   = 10*log10(s2(1:end/2));

figure(1);
plot(fqs1, s1, fqs2, s2);

