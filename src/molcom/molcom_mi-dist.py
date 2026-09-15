# 
# #INFO: Calculation of the MI values of dist for the main plot
# 
# -----------------------------------------------------------------------------
# 
# Author: DTMc
# 
# This file is part of molcom.
# 
# molcom is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# molcom is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# -----------------------------------------------------------------------------

import molcom as mc
import csv
import numpy as np
from pathlib import Path

# The following code prints out the mutual information of various distance rates
# with respect to SNR.

SAVE_CSV_PATH="./datasets/"
Path(SAVE_CSV_PATH).mkdir(parents=True, exist_ok=True)

# -DTMc

# DEFINE LOOP PARAMETERS -------------------------------------------------------

# Define the range in which SNR will be defined
snr = np.arange(10, 50, 1, dtype=int)

# Calculate the Distance 
dist = np.arange(1, 6)

# Define the data array
data = np.zeros((len(snr), len(dist)))

header = np.array(("SNR"))

# MAIN CODE --------------------------------------------------------------------

dist_index = 0

for dist_val in dist:

    snr_index = 0 

    for SNR in snr:

        # We start with defining the initial conditions of the transmission
        _init = {  
                "bit length"        : 1000,
                "Level count"       : 2,
                "Distance"          : 0.5 * dist_val,
                "Advection"         : 0.000001,
                "Diffusion"         : np.array([0.088, 0.088]),
                "Mass"              : 1,
                "RNG"               : np.array([42, 89]),
                "Symbol length"     : 20,
                "SNR"               : SNR,
                "Transmission Type" : "open",
                "Sampling Value"    : 20,
                "Decoding Method"   : "OOK",
            } 

        # We then do the transmission of the chemicals over open space
        m_stream, m_C, m_C_noised = mc.m_molcom_transmission_simulator(
            _init["bit length"], 
            _init["Level count"],
            _init["RNG"],
            _init["Distance"],
            _init["Advection"],
            _init["Diffusion"],
            _init["Symbol length"],
            _init["Mass"],
            _init["SNR"],
            _init["Transmission Type"] 
        )

        # Then we conduct the decoding of the signal of the transmitter
        Tx = mc.m_sampler(
            m_C,
            _init["Symbol length"]
        )

        # Let's not forget the receiver and do the same action on it
        Rx = mc.m_sampler(
            m_C_noised,
            _init["Symbol length"]
        )

        # As the communication relies on ChQAM, we do the 2D decoding of the
        # chemicals
        Decoded_Tx = mc.m_2d_decoder(Tx)        
        Decoded_Rx = mc.m_2d_decoder(Rx)

        # We finally calculate the mutual information
        data[snr_index][dist_index] =  mc.mutual_information(
            Decoded_Tx,
            Decoded_Rx,
            _init["Level count"])

        snr_index += 1

    header = np.append(header, str("x=" + str(0.5 * dist_val)))
        
    dist_index += 1

# Insert results by column    
output = np.c_[snr,data]

# Add the header information to the csv file
output = np.vstack((header, output))

# Save the result to csv
mc.util_save_to_csv(str(SAVE_CSV_PATH + "mi-dist.csv"), output)


#  -----------------------------------------------------------------------------
#  molcom_mi-dist.py ends here.
# 
# 
