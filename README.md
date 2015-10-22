# rm-processing
Programmmed with python

Starts from Aug. 26th, 2015

First upload: Sep. 26th, 2015

Alpha v0.1 launch: Oct. 23rd, 2015

After numerous corrections and debuggings, the rm processing program is now ready to launch the first alpha.

RM stands for reverberation mapping, a common technique used to get the size of broad line area (BLA) of AGN, by finding the time lag between emission lines and continuum. It consists of mainly five steps: read out the data from fits image, fitting continuum and emission lines, integrating the lines and get the light curve for continuum and lines, find lag and plot lag vs continuum. The most difficult and easy to fail one is the second step as different AGNs might behave differently.

# Dependencies:

0. Python 2.7.10 (Not working for Python 3 as psrm has no Python 3 edition)

1. psrm softpackage (Developed by Gao Yang, not downloadable from pypi)

2. numpy >= 1.9.3

3. astropy >= 1.0.4 (Or fitting might not work)

4. scipy >= 0.13.3 (This might not be needed in future release)

5. javelin (https://bitbucket.org/nye17/javelin)

# Other requirements

0. Do not try to run under Windows because os.fork() is used in lc2lag.py!!

1. Recommended running under linux, especially under Ubuntu 14.04

# Running order

fits2data -> data2line -> line2lc -> lc2lag -> lag fit -> correction -> lum vs lag

# Further Improvement

This is only the alpha version and I have actually many improvement to make.First is to modulize this whole thing by introducing Spectrum class, Object class and merge all function into a module. Second, currently only in lc2lag do I utilize multi-thread, which can be easily spread to fits2data and other functions to get a large boost in speed. Third, Hbeta line fit can be wider to include Fe lines inside as super-eddington AGNs have stronger Fe and supressed OIII.
