# 
# #INFO: Calculation of diff values for the main plot
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
snr = np.arange(10, 51, 1, dtype=int)

# Calculate the Distance 
diff = np.arange(1,6)

# Define the data arrays

ber_data = np.zeros((len(snr), len(diff)))

mi_data = np.zeros((len(snr), len(diff)))

cluster_data = np.zeros((3*len(diff), 1000))

header = np.array(("SNR"))

# MAIN CODE --------------------------------------------------------------------

diff_index = 0

patch = [[0],[0]]
 
print("# BEGIN SIMULATION ---")
 
for diff_val in diff:
    
    snr_index = 0

    print("")
    print("# SET diff = " + str(diff_val) + " ---")
    print("")
    
    for SNR in snr:

        print("Currently calculating SNR " + str(SNR)) 

        # We start with defining the initial conditions of the transmission
        _init = {  
                "bit length"        : 1000,
                "Level count"       : 2,
                "Distance"          : 1,
                "Advection"         : 0.000001,
                "Diffusion"         : np.array([0.088 * diff_val, 0.088]),
                "Mass"              : 1,
                "RNG"               : np.array([42, 89]),
                "Symbol length"     : 30,
                "SNR"               : SNR,
                "Transmission Type" : "open",
                "Sampling Value"    : 30, 
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
        Tx = m_stream

        #print(m_stream)
        
        # Let's not forget the receiver and do the same action on it
        Rx = mc.m_sampler(
            m_C_noised,
            _init["Symbol length"]
        )
        
        # As the communication relies on ChQAM, we do the 2D decoding of the
        # chemicals
        Decoded_Tx = mc.m_2d_decoder(Tx)

        # We remove the first bit in the array which was introduced in
        # gen_symbol which was needed to START the simulation and is not
        # part of the simulation.
        Decoded_Rx = np.delete(mc.m_2d_decoder(Rx),0)
        
        if SNR == 30: 

             # Save the transmited bit information
             cluster_data[diff_val - 1][:]  = Rx[0][:]

             # Save the q-axis transmission
             cluster_data[diff_val - 1 + len(diff)][:]  = Rx[1][:]

             cluster_data[diff_val - 1 + 2*len(diff)][:] = \
                 np.insert(np.delete(Decoded_Tx[:],-1,-1),0,0)
             
        
        ber_data[snr_index][diff_index] = \
            mc.BER(Decoded_Rx, Decoded_Tx) / _init["bit length"]

        # We finally calculate the mutual information
        mi_data[snr_index][diff_index] =  mc.mutual_information(
            Decoded_Tx,
            Decoded_Rx,
            _init["Level count"])

        snr_index += 1

    header = np.append(header, str("Da / Db =" + str(diff_val)))
        
    diff_index += 1


print("# END SIMULATION ---")
    
# Insert results by column    
mi_output = np.c_[snr,mi_data]

# Add the header information to the csv file
mi_output = np.vstack((header, mi_output))

# Insert results by column    
ber_output = np.c_[snr,ber_data]

# Add the header information to the csv file
ber_output = np.vstack((header, ber_output))

# # Save the result to csv
# util_save_to_csv(str(SAVE_CSV_PATH + "mi-diff.csv"), mi_output)
# util_save_to_csv(str(SAVE_CSV_PATH + "ber-diff.csv"), ber_output)
mc.util_save_to_csv(str(SAVE_CSV_PATH + "cluster-diff.csv"), np.transpose(cluster_data))

#  -----------------------------------------------------------------------------
#  molcom_diff.py ends here.
# 
# 
