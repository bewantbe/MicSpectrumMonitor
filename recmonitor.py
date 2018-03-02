# Monitor for audio recorder

import time
import array

class shortPeriodDectector:
    """ Detecting a short period in a sequence """

    # _error_thres: used to accept the guessed period
    def __init__(self, _max_period, _cmp_mode = 0, _error_thres = float('inf')):
        self.max_period = int(_max_period)
        self.buffer = array.array('d', [float('nan')]*(2*self.max_period+1))
        self.idx = -1
        #self.period_score = array.array('d', [float('nan')]*self.max_period)
        self.cmp_mode = _cmp_mode
        self.error_thres = _error_thres
        self.period_error = self.error_thres
        self.period_guess = 1

    def bufAt(self, i):
        return self.buffer[(self.idx-i) % len(self.buffer)]

    def bufAppend(self, x):
        self.idx = (self.idx + 1) % len(self.buffer)
        self.buffer[self.idx] = x

    def norm(self, a, n = 1):
        assert(n != 0)
        if n == 1:
            return sum(map(abs, a))
        else:
            return sum(map(lambda x: abs(x)**n, a)) ** (1.0/n)

    def append(self, t_abs):  # input absolute time
        self.bufAppend(t_abs)
        min_s = float('inf')
        min_k = self.period_guess
        for k in range(1, self.max_period+1):
            if self.cmp_mode == 0:
                # detect recurrence of values
                dt = [self.bufAt(j) - self.bufAt(j+k) for j in range(k)]
                s = self.norm(dt, 1)
            elif self.cmp_mode == 1:
                # detect recurrence of increments
                pt = [self.bufAt(j+1) - self.bufAt(j+k+1) for j in range(k)]
                mpt = self.bufAt(0) - self.bufAt(k)
                s = self.norm([pt[j] - mpt for j in range(k)], 1)
            else:
                # detect recurrence of increments, more tolerate to jitter
                pt = [self.bufAt(j) - self.bufAt(j+k) for j in range(k+1)]
                mpt = sum(pt) / (k + 1.0)
                s = self.norm([pt[j] - mpt for j in range(k+1)], 1)
            if s <= min_s:  # '=' for prefer long period
                min_s = s
                min_k = k
        self.period_error = min_s
        self.period_guess = min_k
        return (min_s < self.error_thres, min_k, min_s)

class overrunChecker:
    """ Check audio buffer for returning less data then expected """
    # time unit is second
    def __init__(self, _sample_rate, _buffer_sample_size):
        self.time_update_old = 0
        self.time_update_interval = 2.0
        self.time_started = 0
        self.last_overrun_time = 0
        self.n_total_samples = 0
        self.sample_rate = _sample_rate
        self.buffer_sample_size = _buffer_sample_size
        self.sample_rate_est = _sample_rate
        self.last_check_overrun = False
        # private varialbes
        self.n_total_samples_old = self.n_total_samples
        self.time_now_old = 0
        self.__t_old = 0

    def start(self):
        self.n_total_samples = 0
        self.last_overrun_time = 0
        self.time_started = time.time()
        self.time_update_old = self.time_started
        self.sample_rate_est = self.sample_rate

    def printStat(self, n_samples_from_time):
        f1 = n_samples_from_time / self.sample_rate_est
        f2 = self.n_total_samples / self.sample_rate_est
        print("  Time: %s" % time.strftime("%F %T", time.gmtime(time.time())))
        print("  Should read %.0f = %.3f sec." % (n_samples_from_time, f1))
        print("  Actual read %.0f = %.3f sec." % (self.n_total_samples, f2))
        print("  Difference  %.0f = %.3f sec." % \
                (n_samples_from_time - self.n_total_samples, f1-f2))
        print("  Estimated sample rate = %.2f Hz, (requested: %.0f Hz)" % \
                (self.sample_rate_est, self.sample_rate))

    def updateState(self, n_read_samples):
        time_now = time.time()
        dt_s = time_now - self.__t_old
        self.__t_old = time_now

        if self.n_total_samples == 0:
            self.time_started = time_now - n_read_samples / self.sample_rate
        self.n_total_samples += n_read_samples

        if time_now < self.time_update_old + self.time_update_interval \
          or dt_s < 0.001:
            return

        sr_intv = (self.n_total_samples - self.n_total_samples_old) / (time_now - self.time_now_old)
        print("")
        print("  intv sr: %.1f Hz" % (sr_intv))
        print("   est sr: %.1f Hz" % (self.sample_rate_est))
        print("   ave sr: %.1f Hz" % (self.n_total_samples / (time_now - self.time_started)))

        # Check if recorder buffer overrun occur.
        n_samples_from_time = (time_now - self.time_started) * self.sample_rate_est
        if n_samples_from_time >= 2*self.buffer_sample_size + self.n_total_samples:
            print("Suspect recorder buffer overrun!")
            self.printStat(n_samples_from_time)
            self.last_overrun_time = time_now
            self.n_total_samples = 0
            print("Overrun counter reset.\n")

        # Update sampling rate estimation.
        if self.n_total_samples >= 10 * self.sample_rate:
            self.sample_rate_est = 0.9 * self.sample_rate_est + \
              0.1 * (self.n_total_samples / (time_now - self.time_started))
            if abs(self.sample_rate_est / self.sample_rate - 1) > 0.0145:
                print("Sample rate significant deviation!")
                self.printStat(n_samples_from_time)
                self.n_total_samples = 0
                print("Overrun counter reset.")

        self.last_check_overrun = self.last_overrun_time == time_now
        
        self.time_update_old += self.time_update_interval
        # Catch up with current time, so that at most one output per call.
        if self.time_update_old + self.time_update_interval <= time_now:
            self.time_update_old = time_now  
        self.n_total_samples_old = self.n_total_samples
        self.time_now_old = time_now

    def getLastCheckOverrun(self):
        return self.last_check_overrun

    def getLastOverrunTime(self):
        return self.last_overrun_time

    def getSampleRate(self):
        return self.sample_rate_est

