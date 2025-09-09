import numpy as np
import os
import scipy.interpolate as interp
import scipy.optimize as sciopt

debug = False



class Extractor():
    def __init__ (self):
        pass

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout, ibias = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)
        ugbw = self.find_ugbw(freq, vout) / 1e6
        phm = self.find_phm(freq, vout)

        spec = dict(ugbw=ugbw, gain=gain, phm=phm, ibias=ibias)

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, "ac.csv")
        dc_fname = os.path.join(output_path, "dc.csv")

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j * vout_imag
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain(self, vout):
        # return np.abs(vout)[0]
        return 20*np.log10(np.abs(vout)[0])
    

    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        if valid:
            return ugbw
        else:
            return freq[0]

    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase)  # unwrap the discontinuity
        phase = np.rad2deg(phase)  # convert to degrees
        #
        # plt.subplot(211)
        # plt.plot(np.log10(freq[:200]), 20*np.log10(gain[:200]))
        # plt.subplot(212)
        # plt.plot(np.log10(freq[:200]), phase)

        phase_fun = interp.interp1d(freq, phase, kind="quadratic")
        ugbw, valid = self._get_best_crossing(freq, gain, val=1)
        print ("phase_fun(ugbw): ", phase_fun(ugbw))
        if valid:
            if phase_fun(ugbw) > 0:
                return -180 + phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            return -180

    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop), True
        except ValueError:
            # avoid no solution
            # if abs(fzero(xstart)) < abs(fzero(xstop)):
            #     return xstart
            return xstop, False

if __name__ == "__main__":
    extractor = Extractor()
    print (extractor.translate_result("."))