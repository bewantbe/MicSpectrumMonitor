% time aligment

addpath('/home/xyy/matcode/stft_mrfft/');
x1=[];
if ~exist('x1', 'var') || isempty(x1)
    fname1 = 'c51_imm6_vol20000.wav';
    fname2 = 'c52_umik1_vol65536.wav';

    fprintf('reading ...');fflush(stdout); tic;
    [x1 sr1] = wavread(fname1);
    [x2 sr2] = wavread(fname2);
    x1 = x1(:,1);
    x2 = x2(:,1);
    fprintf(' done. (t = %.3f sec)\n', toc);fflush(stdout);
end

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

rg = (1:1e6) + 1e6;
gain1 = 10^(3.5/10);

figure(11); xf1 = show_stft(x1(rg)*gain1, 48e3);
figure(12); xf2 = show_stft(x2(rg), 48e3);

f_error_level = @(x) quantile(x(:), 0.5);

p2db = @(x) 10*log10(x);
reL = @(x) max(x, 0);

c = 0;
for iv = 2:size(xf1,1)     % number of frequency points
    v1 = p2db(abs(xf1(iv, :)) .^ 2);
    v2 = p2db(abs(xf2(iv, :)) .^ 2);
    v1 = reL(v1 - f_error_level(v1) - 5);
    v2 = reL(v2 - f_error_level(v2) - 5);

    v1 = expm1(0.5*log(10)*(v1/10));
    v2 = expm1(0.5*log(10)*(v2/10));

    c += real(ifft(fft(v1) .* conj(fft(v2))));
end

figure(15);
cid = [0:floor((length(c)+1)/2) (floor((length(c)+1)/2)+1:length(c)-1) - length(c)];
plot(fftshift(cid), fftshift(c), '-*');
xlim([-50,50]);

iv = round(size(xf1,1)/2/5);
    v1 = p2db(abs(xf1(iv, :)) .^ 2);
    v2 = p2db(abs(xf2(iv, :)) .^ 2);
    v1 = reL(v1 - f_error_level(v1) - 5);
    v2 = reL(v2 - f_error_level(v2) - 5);

    v1 = expm1(0.5*log(10)*(v1/10));
    v2 = expm1(0.5*log(10)*(v2/10));
figure(16);
plot(v1)
figure(17);
plot(v2)


% vim: set expandtab shiftwidth=4 softtabstop=4:
