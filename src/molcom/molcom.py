# 
# #INFO: A macro-scale molecular communication library
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

"""
The pythonic module to calculate and analyze molecular-communication using
a wide variety of communication methods. This module allows the simulation of
the physical layer of the macro-scale molecular communication

(DTMc, 2025) v0.87
"""


import numpy as np
from scipy import special
from itertools import groupby
import csv
 

def gen_symbol(length_val, level, seed_val=None):
    """Generates a random stream of symbols based on a given length. To
    aid in its reproducability, a seed value can be given. If no seed value
    is given the program will randomly generate seed value based on what it
    sees fit.

    References
    ----------
    [1] https://docs.python.org/3/library/itertools.html#itertools.groupby
    [2] https://stackoverflow.com/questions/39340345/how-to-count-consecutive-duplicates-in-a-python-list

    Parameters
    ----------
    length_val : (int) the length of the transmission in symbols.
    level      : (int) the number of levels the transmission support.
    seed_val   : (int) the RNG to enter for reproducability.

    Returns (in order)
    -------
    stream     : (np.array) the stream of symbols to transmit.
    sym_list   : (np.array) the list of symbols in order.
    sym_freq   : (np.array) the successive occurence of each symbol.
    
    Examples
    --------
    stream, sym_list, sym_freq = gen_symbol(10, 2, 42)
    """

    if seed_val is not None:
        # set the seed for the RNG
        np.random.seed(seed=seed_val)

    # generate the random stream of symbols
    stream = np.random.randint(level, size=length_val)

    # generate the unique symbol list
    sym_list = np.array([k for k, g in groupby(stream)])

    # generate the unique symbol frequency list
    sym_freq = np.array(
        [sum(1 for _ in group) for _, group in groupby(stream)])

    return stream, sym_list, sym_freq


def __transmission_function(dist, time, advection, diffusion, method):
    """A PRIVATE function to generate the non-elementary functions to
    calculate the mass transportation along a certain environment with
    specific boundary conditions.

    Parameters
    ----------
    dist      : float
        Transmission distance between the transmitter and the receiver
        [unit: m]
    time      : int
        Time value in which the function generates data
        [unit s]
    advection : float
        The advective flow (constant) present in the environment
        [unit m/s]
    diffusion : float
        The diffusion coefficient of the transmitted chemical
        [unit m2/s]
    method : string
        The environment in which the gas is propagating.
        Supported options are:
        - "open" : transmission with no boundaries and hinderances.

    Examples
    --------
    This is a PRIVATE function not meant to be used by the user.

    """

    # Returns the open transmission function
    if method == "open":
        return (special.erf((dist-time*advection)\
                            /(2*np.sqrt(diffusion*time))) \
                + special.erf((dist+time*advection)\
                              /(2*np.sqrt(diffusion*time))))/2


def __transmission_duration(dist, duration, advection, diffusion):
    """Calculates the transmission function

    Parameters
    ----------
    dist : float
        Transmission distance [unit m]
    duration : int
        The array length for the function
    advection : float
        The advective flow of the transmission [unit m/s]
    diffusion : float
        The diffusive coefficeint of the transmitted chemical [unit m2/s]

    Examples
    --------
    This is a PRIVATE function not meant to be used by the user.

    """

    # Generate an empty array to store the results
    ADE_func = np.zeros(duration)

    # Loop through and assing the calculated values to the array
    for i in range(1, duration + 1, 1):
        ADE_func[i - 1] = __transmission_function(dist, i, advection,
                                                  diffusion, "open")

    # Retuns the calculated values
    return ADE_func


def molcom_transmission_simulator(dist,
                                  advection,
                                  diffusion,
                                  symbol_length,
                                  sym_list,
                                  sym_freq,
                                  mass,
                                  SNR,
                                  method=None):
    """Function to simulate macro-molecular communcations

    Parameters
    ----------
    dist : float
        Transmission distance [unit m/s]
    advection : float
        advective float of the transmission
    diffusion : float
        The diffusive coefficient of the gas [unit m2/s]
    symbol_length : int
        The symbol length of an individual symbol [unit s]
    sym_list : np.array
        A list of unique symbols in sequence
    sym_freq : np.array
        The frequency of transmitted symbols in sequence
    mass : float
        The mass of the transmitted chemicals
    SNR : float
        The SNR of the received signal
    method : 
        The specific boundary conditions for the transmission

    Examples
    --------
    molcom_transmission_simulator(0.1, 0.01, 0.0124, 20, sym_list, sym_freq, 1, 50, "open")

    """

    # First generate a loop function to pre-calculate the function
    __loop_func = __transmission_duration(dist,
                                          np.max(sym_freq) * symbol_length,
                                          advection, diffusion)

    # Designate the initial mass
    C = [1]

    # Loop throught the sym_list values
    for j in range(1, len(sym_list)):

        # A special case for the first bit
        if j == 1:

            # If it is 0 append just zeros until a non-zero bit is sent
            if sym_list[j - 1] == 0:

                C = np.append(C, np.zeros(sym_freq[j - 1] * symbol_length))

            # Else just do an initial gas transmission
            else:

                C_p = sym_list[j-1] \
                    * (1 - __loop_func[0:sym_freq[j-1] * symbol_length])

                C = np.append(C, C_p)

        # Do the loop and append C array based on the values on the
        # sym_list
        if sym_list[j] > sym_list[j - 1]:

            C_p = C[-1] + (sym_list[j] - sym_list[j-1]) \
                * (1 - __loop_func[0:sym_freq[j] * symbol_length])

            C = np.append(C, C_p)

        else:

            C_n = (np.abs(C[-1] - sym_list[j]))\
                * __loop_func[0:sym_freq[j]* symbol_length] \
                + sym_list[j]

            C = np.append(C, C_n)

    # Calculating the SNR of the transmitted signal
    C_power = np.power(C, 2)  # define the power of the signal

    target_snr_db = SNR  # designate SNR target

    # Calculate the average watts & dB of the signal
    sig_avg_watts = np.mean(C_power)
    sig_avg_db = 10 * np.log10(sig_avg_watts)

    # Calculate the average watts & dB of the noise
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10**(noise_avg_db / 10)

    # Define the noise variables
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),
                                   len(C_power))

    C_garble = C + noise_volts

    # Return the transmitted signal and the signal with noise.
    return np.delete(C * mass, 0), np.delete(C_garble * mass, 0)


def sampler(signal, sampling):
    """A simple function to down sample the received signal

    Parameters
    ----------
    signal : np.array
        Received signal array
    sampling : int
        The sampling value

    Examples
    --------
    sampler(C_noised, 20)


    """
    return signal[1::sampling]


def shannon_log(value):
    """A simple function to simulate shannon's log2 function

    Parameters
    ----------
    value : float
        can be any numerical value

    Examples
    --------
    A PRIVATE function not meant for end user.
    
    __shannon_log(2)

    """    
    # Return 0 if the given value is 0
    
    if value == 0:
        return 0
    else:
        try:
            return np.log2(value)
        except ZeroDivisionError:
            return 0


def generate_received_matrix(transmit_bits, received_bits, level):
    """Generate the received symbol matrix

    Parameters
    ----------
    transmit_bits : np.array
        An array of transmitted (pre disturbed) bits
    received_bits : np.array
        An array of received (post disturbed) bits
    level : int
    symbols used in the transmission

    Examples
    --------
    A PRIVATE function not meant for end user.
    
    __generate_received_matrix(stream, Decoded, 2)


    """

    # Create an empty 2D array based on symbol count
    receiver = np.zeros((level*level, level*level))

    # Loop and add each occured value
    for i in range(len(received_bits)):
        receiver[int(transmit_bits[i]), int(received_bits[i])] += 1

    return receiver / len(received_bits)


def received_bit_classifier(sampled_signal, method):
    """Classify the received analog value based on chosen method

    Parameters
    ----------
    sampled_signal : np.array
        received signal value as an array
    method : string
        The method used in classifying the bits

    Examples
    --------
    FIXME: Add docs.

    """

    Decoded = np.array([])

    if method == "OOK":

        for val in sampled_signal:
            if val >= 0.5:
                Decoded = np.append(Decoded, 1)
            else:
                Decoded = np.append(Decoded, 0)

    return Decoded


def m_molcom_transmission_simulator(bit_length,
                                    level,
                                    RNG,
                                    distance,
                                    advection,
                                    diffusion,
                                    symbol_duration,
                                    mass,
                                    SNR,
                                    method=None):
    """Create multiple chemical transmission which are independent from each other

    The function creates a simulation of a transmission using multiple
    chemicals occupying the same channel. The transmsision is assumed
    to be independent an non-interfering with each other. The function
    returns the intended transmitted bit-stream, the transmitted
    chemical signals with no noise and with noise.

    Parameters
    ----------
    bit_length : int
        The number of bits (or symbols) wanted to transfer
    level : int
        Different types of symbols to transmit (i.e., 0 and 1 would be
        2)
    RNG : np.array
        The seed to generate the bit-stream
    distance : np.array
        Transmission distance [m]
    advection : float
        The advective flow of the tranmission [m/s]
    diffusion : float
        The diffusive coefficient of the transmitted chemical [m2/s]
    symbol_duration : int
        The time given for the transmission of ONE symbol
    mass : float
        the mass of a unit symbol
    SNR : float
        The SNR value of the desired signal output
    method : string
        The method in which the simulation takes place

    Examples
    --------
    _init = {
    "bit length": 100,
    "Level count": 2,
    "Distance": 0.1,
    "Advection": 0.01,
    "Diffusion": [0.0124 0.0168],
    "Mass": 1,
    "Symbol length": 20,
    "SNR": 50,
    "Transmission Type": "open",
    "Sampling Value": 20,
    "Decoding Method": "OOK",
    }

    m_stream, m_C, m_C_noised = m_molcom_transmission_simulator(
    _init["bit length"],
    _init["Level count"],
    RNG,
    _init["Distance"],
    _init["Advection"],
    _init["Diffusion"],
    _init["Symbol length"],
    _init["Mass"],
    _init["SNR"],
    _init["Transmission Type"])

    """

    # Define the multi channel arrays
    m_stream = np.empty((0, bit_length), int)  # The intended bit stream
    m_C = np.empty((0, bit_length * symbol_duration))  # The received signal
    m_C_noised = np.empty(
        (0, bit_length * symbol_duration))  # The signal with noise

    # Loop through based on the RNG array.
    for _ in range(len(RNG)):

        # Generate the stream and simulation dependant arrays
        stream, sym_list, sym_freq = gen_symbol(bit_length, level, RNG[_])

        # Do the simulation for each channel independently
        C, C_noised = molcom_transmission_simulator(distance, advection,
                                                    diffusion[_], symbol_duration,
                                                    sym_list, sym_freq, mass,
                                                    SNR, method)

        # Save results into arrays as rows
        m_stream = np.append(m_stream, [stream], axis=0)
        m_C = np.append(m_C, [C], axis=0)
        m_C_noised = np.append(m_C_noised, [C_noised], axis=0)

    # Return the values to user
    return m_stream, m_C, m_C_noised

    
def m_sampler(signal, sampling):
    """A simple function to down sample the received signal of a multi
    channel transmission.

    Parameters
    ----------
    signal : np.array
        Received signal array
    sampling : int
        The sampling value.
        NOTE -- set it to symbol length.

    Examples
    --------    
    sampler(C_noised, 20)

    """

    # Generate empty matrix to store the sampled values
    m_sampled_signal = np.empty((0, int(signal.shape[1] / sampling)))

    # Iterate over the rows and use the 1D function on each one
    for row in signal:
        m_sampled_signal = np.append(m_sampled_signal,
                                     [sampler(row, sampling)],
                                     axis=0)

    # Return the calculated 2D array
    return m_sampled_signal


def entropy(probability):

    return -probability * shannon_log(probability)


def mutual_information(input_alphabet, output_alphabet, level):
    """Calculates the mutual information

    Measures the mutual information between the input and the output
    alphabet. It uses HX - HXY to calculate

    Parameters
    ----------
    input_alphabet : np.array
        The input alphabet
    output_alphabet : np.array
        The received alphabet
    level : int
        the modulated level of the signal

    Examples
    --------
    mutual_information(Decoded_Tx, Decoded_Rx, 2)

    """

    # Generate the matrix of received symbols
    Receiver = generate_received_matrix(input_alphabet, output_alphabet, 2)
    
    # Calculate the probabilities
    PX = np.sum(Receiver, axis=1)  # vectical is X
    PY = np.sum(Receiver, axis=0)  # horizon is Y
    
    # Calculate the entropies
    HY = sum([entropy(p) for p in PY])
    HX = sum([entropy(p) for p in PX])

    # Define the conditional entropy
    HXY = np.zeros((level*level,level*level))

    for i in range(0, level*level):
        for j in range(0, level*level):
            HXY[i][j] =  - Receiver[i, j] * shannon_log(Receiver[i, j] / PX[i])
            
    return HX - HXY.sum() 


def util_save_to_csv(filename, variable):
    """Saves results to csv

    The command takes two arguments. First the filepath, and then the dataset. Then it saves
    the files to csv. If a file exists, then it overwrites it.

    Parameters
    ----------
    filename : PATH
        the path of the csv file to be saved.
    variable : np.array
        The dataset to be saved

    Examples
    --------
    util_save_to_csv("~/Downloads/scatter-chem.csv", data)

    """
    
    with open(filename,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(variable)



def BER(decoded_rx, decoded_tx):
    """Calculates the BER of the communcation.

    This function takes the decoded transmission of the transmitted and the
    received and does a bit-by-bit comparison to calculate the bit-error-rate
    (BER) of the communication.

    References
    ----------
    [1] https://stackoverflow.com/questions/55986094/comparing-two-lists-element-wise-in-python

    Parameters
    ----------
    decoded_rx : list
        The list containing the decoded bits of the communication from the
        receiver end.
    decoded_tx : list
        The list containing the decoded bits of the communication from the
        transmitter end.

    Examples
    --------
    BER(Decoded_Rx, Decoded_Tx)

    """

    # Define the error variable which will be added per each error found.
    err = 0

    # Loop through the list. Join them using a touple to make it more pythonic!
    for tx, rx in zip(decoded_tx, decoded_rx):
        
        if tx != rx:
            err += 1

    return err


# EOC --------------------------------------------------------------------------

def m_2d_decoder(signal):

    Decoded = np.array([])

    for i in range(len(signal[0,:])):
        if signal[0,i] < 0.5 and signal[1,i] < 0.5:
            Decoded = np.append(Decoded, 0)
        elif signal[0,i] >= 0.5 and signal[1,i] >= 0.5:
            Decoded = np.append(Decoded, 1)
        elif signal[0,i] < 0.5 and signal[1,i] >= 0.5:
            Decoded = np.append(Decoded, 2)
        elif signal[0,i] >= 0.5 and signal[1,i] < 0.5:
            Decoded = np.append(Decoded, 3)

    return Decoded

# ------------------------------------------------------------------------------
# molcom.py ends here.
# 
# 
