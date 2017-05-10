% cpm spectrum

% compare sampling rate
%{
sleep 2
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 44100 -c 1 --duration=10 a1.wav
sleep 1
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 a2.wav
sleep 1
arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 96000 -c 1 --duration=10 a3.wav
%}

% compare different mic
%{
sleep 3; arecord -vv --dump-hw-params -D 'hw:CARD=U18dB' -f S24_3LE -r 48000 -c 2 --duration=10 b1.wav
sleep 1; arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b2.wav
sleep 3; arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b3.wav
sleep 3; arecord -vv --dump-hw-params -D 'default' -f S16_LE -r 48000 -c 1 --duration=10 b4.wav
%}

%fname1 = 'rec2017-05-09_05h43m39.086s_hw.wav';
%fname2 = 'c_c1.wav';
%fname1 = 'rec2017-05-09_04h28m43.710s_sm.wav';
%fname2 = 'c_c1.wav';
%fname1 = 'b22_xde_umik1_id3270_vol26090.wav';
%fname2 = 'b21_xde_imm6_id8_vol12400.wav';

%fname1 = 'c02_xde_umik1_id3270_vol26090.wav';
%fname2 = 'rec2017-05-09_12h20m59.545s_sm_imm6-id4.wav';

fname1 = 'c51_imm6_vol20000.wav';
fname2 = 'c52_umik1_vol65536.wav';

fprintf('reading ...');fflush(stdout); tic;
[x1 sr1] = wavread(fname1);
[x2 sr2] = wavread(fname2);
x1 = x1(:,1);
x2 = x2(:,1);

% time aligment
shift1 = floor(1024/4 * -446.815);
if shift1 >= 0
  x1 = x1(1:end-shift1);
  x2 = x2(1+shift1:end);
else
  shift1 = -shift1;
  x1 = x1(1+shift1:end);
  x2 = x2(1:end-shift1);
end

fprintf(' done. (t = %.3f sec)\n', toc);fflush(stdout);

%x1 = x1 * (10 ^ (2.4/10));  % cell phone adjust to match volume 9000

x1 = x1 * (10 ^ (3.5/10));

x1 = x1(round(end/3):round(end*2/3));
x2 = x2(round(end/3):round(end*2/3));

%whitening_od = 3;
%fprintf('filtering ...');fflush(stdout); tic;
%%x1 = x1_old; % = x1;
%[Aall, Deps] = ARregression(getcovpd(x1', whitening_od));
%x1 = filter([1, Aall], [1], x1.' - mean(x1)).';
%x2 = filter([1, Aall], [1], x2.' - mean(x2)).';

%%x2 = x2_old; % = x2;
%%[Aall, Deps] = ARregression(getcovpd(x2', whitening_od));
%%x2 = filter([1, Aall], [1], x2.' - mean(x2)).';
%fprintf(' done. (t = %.3f sec)\n', toc);fflush(stdout);

rms1 = 20*log10(sqrt(x1' * x1 * 2 / length(x1)));
rms2 = 20*log10(sqrt(x2' * x2 * 2 / length(x2)));
fprintf('rms1 = %5.2f dB\n', rms1);
fprintf('rms2 = %5.2f dB\n', rms2);

addpath('/home/xyy/matcode/stft_mrfft/');

sz_dat = 2^10;

f_wnd = @(x) 0.5+0.5*cos(2*pi*x);  % hanning
wnd = f_wnd(((1:sz_dat).'-0.5)/sz_dat - 0.5);

fprintf('sft ...');fflush(stdout); tic;
[s1, fqs1] = sft_wnd_c(x1, wnd, [], [], 'audio');
fqs1 = fqs1(1:end/2) * sr1;
s1   = 10*log10(s1(1:end/2));
fprintf(' done 1 (t = %.3f sec)\n', toc);fflush(stdout); tic
[s2, fqs2] = sft_wnd_c(x2, wnd, [], [], 'audio');
fqs2 = fqs2(1:end/2) * sr2;
s2   = 10*log10(s2(1:end/2));
fprintf('    ... done 2 (t = %.3f sec)\n', toc);fflush(stdout);

figure(1);
plot(fqs1, s1, fqs2, s2);
h = legend(fname1, fname2);
set(h, 'Interpreter', 'none');

figure(2);
semilogx(fqs1(2:end), s1(2:end), fqs2(2:end), s2(2:end));
h = legend(fname1, fname2);
set(h, 'Interpreter', 'none');

