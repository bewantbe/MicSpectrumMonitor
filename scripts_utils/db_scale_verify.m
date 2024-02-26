%

n = 1024;
m = 10000;

%X = 1/32768 * (rand(n,m) - 0.5);
%X = 2 * (rand(n,m) - 0.5);          % RMS = 10*log10(1/3) dB
%X = ones(n,m);
X = sin((0:n-1)/n * 100*pi).';
%X = sign(sin((0:n-1)/n * 100*pi).');
%X = sign(sin(0.0001 + (0:n-1)/n * 128*pi).');  # exact period

eng    = mean(X(:).^2)
rms_db = 10*log10(eng)
rms_db_theory_rand = 20*log10(1/sqrt(3)/2^16)

wnd = 0.5*cos(((0:n-1)/n-0.5).' * 2*pi)+0.5;
%wnd = ones(n,1);

wnd_db_rel = 20*log10(sum(wnd) / n)

fqs = (0:n-1)/n;

sp_normalization = 'energy'
switch sp_normalization
  case 'sine_amplitude'
    % such that 1.0*sin(wt) (w!=0) will give 0dB spectrum peak
    % meanwhile, RMS = 10*log10(1/2) dB = -3.01 dB
    wnd = wnd / sum(wnd) * 2;     % "* 2" due to nyquist
  case 'square_rms'
    % such that the sum of all theoretical fft peaks in 1.0*sign(sin(wt)) (w!=0) 
    % will give 0dB, which is the same as RMS.
    wnd = wnd / sum(wnd) * 2 / sqrt(2);
    % best match with wnd = ones(n,1), with window, the effective window size
    % will decrease, hence higher noise.
    u_theory_simple = rms_db + 10*log10(2/n)
  case 'energy'
    % such that the sum of energy in all frequencies will give RMS.
    % usually used for stochastic signal analysis, more "physical".
    % e.g. used when you need to compare noise spectrum with different
    % sampling rate.
    sr = 1.0;
    wnd = wnd / sqrt(sum(wnd.^2) * sr/2);  % "sr/2" due to nyquist
    fqs = sr * fqs;
end

Y = fft(wnd .* X);

u_theory = 10*log10(sum(wnd.^2)) + 20*log10(1/sqrt(3)/2^16)

u = 10*log10(mean(Y .* conj(Y), 2));

figure(20)
plot(fqs(1:end/2), u(1:end/2))


# measured:



