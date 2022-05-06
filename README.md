# Seismic-optimization
This repository consists of source code for the research paper : "Performance Optimization of Seismic Event Detection Techniques for Earthquake Early Warning System"

# Dataset used
1.Derived database of 10 years Japan earthquake data containing P and S wave ground truth prepared from K-NET data set (https://www.kyoshin.bosai.go.jp/). More information about the database is available at Mendeley Open source data respository (https://dx.doi.org/10.17632/s7rk7bj3zn.1).
Seismic Noise data is collected from National Capital Region, India. More information is available at Mendeley Open source data respository (https://dx.doi.org/10.17632/4wv9pff6b3.1).

# Code
For threshold and leading window duration optimization of MER, MWA, and STALTA run mer_optimize.py, mwa_optimize.py, and stalta_optimize.py from Optimization folder.

After optimization use the optimized values to find the detection perfromance.

Run mer_error_all.py, mwa_error_all.py, and stalta_error_all.py  from the folder Optmized Performance using the optimized value pair to find the detection error of the three techniques.

Run Accuracy.py & Delay_detection.py to calculate detection performance and delay in detection.

# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details
