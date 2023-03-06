# 5G NR PHY frame class

import numpy as np
from dataclasses import dataclass


class PhyLayerFrameStructure:
  
  """
  A class representing the physical (PHY) layer frame structure for 5G NR OFDM numerology combinations.

  Parameters
  ----------
  freq_bandwidth : tuple
    A tuple containing the frequency and bandwidth in MHz (e.g., (3500, 100)).
  numerology : int
    The numerology of the OFDM system (e.g., 0, 1, 2, 3, or 4).
  
  Attributes
  ----------
  freq_bandwidth : tuple
    A tuple containing the frequency and bandwidth in Hz.
  numerology : int
    The numerology of the OFDM system.
  subcarrier_spacing : int
    The subcarrier spacing in Hz.
  cyclic_prefix_length : float
    The length of the cyclic prefix in seconds.
  sampling_rate : int
    The sampling rate in Hz.
  fft_size : int
    The size of the FFT.
  scs_per_rb : int
    The number of subcarriers per resource block.
  rb_size : int
    The size of the resource block.
  num_rbs : int
    The number of resource blocks in the system.
    """

  def __init__(self, freq_bandwidth, numerology):
    self.freq_bandwidth = freq_bandwidth * 10**6
    self.numerology = numerology # 0, 1, 2, 3, or 4

    # Calculate subcarrier spacing and cyclic prefix length
    if numerology == 0:
      self.subcarrier_spacing = 15 * 10**3
      self.cyclic_prefix_length = 16 * 10**-6
    elif numerology == 1:
      self.subcarrier_spacing = 30 * 10**3
      self.cyclic_prefix_length = 16 * 10**-6
    elif numerology == 2:
      self.subcarrier_spacing = 60 * 10**3
      self.cyclic_prefix_length = 16 * 10**-6
    elif numerology == 3:
      self.subcarrier_spacing = 120 * 10**3
      self.cyclic_prefix_length = 16 * 10**-6
    elif numerology == 4:
      self.subcarrier_spacing = 240 * 10**3
      self.cyclic_prefix_length = 16 * 10**-6
    else:
      raise ValueError("Invalid numerology value")

    # Calculate sampling rate, FFT size, subcarriers per RB, RB size, and number of RBs
    self.sampling_rate = 2 * self.freq_bandwidth
    self.fft_size = 2**np.ceil(np.log2(self.sampling_rate/self.subcarrier_spacing))
    self.scs_per_rb = 12 * (self.fft_size // 2048)
    self.rb_size = self.scs_per_rb * self.subcarrier_spacing
    self.num_rbs = np.ceil(self.freq_bandwidth / self.rb_size)

  def get_re_indices(self, rb_idx):
    """
    Returns the indices of the resource elements in a given resource block.

    Parameters
    ----------
    rb_idx : int
        The index of the resource block.

    Returns
    -------
    re_indices : numpy.ndarray
        An array of the indices of the resource elements in the resource block.
    """
    return np.arange(rb_idx * self.scs_per_rb, (rb_idx + 1) * self.scs_per_rb)
  


class NRPHYFrame:
    def __init__(self, numerology, num_slots):
        """
        Initialize a new 5G NR PHY frame with the given numerology and number of slots.
        """
        self.numerology = numerology
        self.num_slots = num_slots
        self.slot_duration = 1 / (2 ** self.numerology)
        self.symbol_duration = self.slot_duration / 14
        self.num_symbols_per_slot = 14 * (2 ** self.numerology)
        self.num_symbols = self.num_symbols_per_slot * self.num_slots
        self.num_subcarriers = 12 * (2 ** self.numerology)
        self.num_sc_per_rb = 12
        self.num_rbs_per_slot = self.num_subcarriers // self.num_sc_per_rb
        self.num_rbs = self.num_rbs_per_slot * self.num_slots
        self.num_re_per_rb = 14
        self.num_re_per_slot = self.num_re_per_rb * self.num_rbs_per_slot
        self.num_re = self.num_re_per_slot * self.num_slots
    
    def get_slot_duration(self):
        """
        Get the duration of a single slot in seconds.
        """
        return self.slot_duration
    
    def get_symbol_duration(self):
        """
        Get the duration of a single symbol in seconds.
        """
        return self.symbol_duration
    
    def get_num_symbols_per_slot(self):
        """
        Get the number of symbols per slot.
        """
        return self.num_symbols_per_slot
    
    def get_num_symbols(self):
        """
        Get the total number of symbols in the frame.
        """
        return self.num_symbols
    
    def get_num_subcarriers(self):
        """
        Get the total number of subcarriers in the frame.
        """
        return self.num_subcarriers
    
    def get_num_rbs_per_slot(self):
        """
        Get the number of resource blocks per slot.
        """
        return self.num_rbs_per_slot
    
    def get_num_rbs(self):
        """
        Get the total number of resource blocks in the frame.
        """
        return self.num_rbs
    
    def get_num_re_per_rb(self):
        """
        Get the number of resource elements per resource block.
        """
        return self.num_re_per_rb
    
    def get_num_re_per_slot(self):
        """
        Get the total number of resource elements per slot.
        """
        return self.num_re_per_slot
    
    def get_num_re(self):
        """
        Get the total number of resource elements in the frame.
        """
        return self.num_re
    
    def get_re_frequency(self, re_idx):
        """
        Get the frequency (subcarrier index) of the given resource element index.
        """
        rb_idx = re_idx // self.num_re_per_rb
        sc_idx = (re_idx % self.num_re_per_rb) % self.num_sc_per_rb
        return rb_idx * self.num_sc_per_rb + sc_idx
    
    def get_re_time(self, re_idx):
        """
        Get the time (symbol index) of the given resource element index.
        """
        slot_idx = re_idx // self.num_re_per_slot
        sym_idx = (re_idx % self.num_re_per_slot) % self.num_symbols_per_slot
        return slot_idx * self.symbol_duration * self.num_symbols_per_slot + sym_idx * self.symbol_duration
    
    def get_re_frequency_and_time(self, re_idx):
        """
        Get the frequency and time of the given resource element index.
        """
        return self.get_re_frequency(re_idx), self.get_re_time(re_idx)
    
    def get_re_frequency_and_time_range(self, re_idx, num_re):
        """
        Get the frequency and time range of the given resource element index and number of resource elements.
        """
        re_freq, re_time = self.get_re_frequency_and_time(re_idx)
        re_freq_range = (re_freq, re_freq + num_re)
        re_time_range = (re_time, re_time + num_re * self.symbol_duration)
        return re_freq_range, re_time_range
    
    def get_re_frequency_range(self, re_idx, num_re):
        """
        Get the frequency range of the given resource element index and number of resource elements.
        """
        return self.get_re_frequency_and_time_range(re_idx, num_re)[0]
    
    def get_re_time_range(self, re_idx, num_re):
        """
        Get the time range of the given resource element index and number of resource elements.
        """
        return self.get_re_frequency_and_time_range(re_idx, num_re)[1]
    
    def get_re_idx(self, re_freq, re_time):
        """
        Get the resource element index of the given frequency and time.
        """
        slot_idx = int(re_time / self.slot_duration)
        sym_idx = int((re_time % self.slot_duration) / self.symbol_duration)
        rb_idx = int(re_freq / self.num_sc_per_rb)
        sc_idx = int(re_freq % self.num_sc_per_rb)
        return slot_idx * self.num_re_per_slot + rb_idx * self.num_re_per_rb + sym_idx * self.num_sc_per_rb + sc_idx
    
def main(numerology, num_slots):
    """
    Main function.
    """
    # Create a new 5G NR PHY frame object
    nr_phy_frame = NRPHYFrame(numerology, num_slots)
    
    # Print the number of resource elements in the frame
    print("Number of resource elements in the frame: {}".format(nr_phy_frame.get_num_re()))
    
    # Print the frequency of the 100th resource element in the frame
    print("Frequency of the 100th resource element in the frame: {}".format(nr_phy_frame.get_re_frequency(100)))
    
    # Print the time of the 100th resource element in the frame
    print("Time of the 100th resource element in the frame: {}".format(nr_phy_frame.get_re_time(100)))
    
    # Print the frequency and time of the 100th resource element in the frame
    print("Frequency and time of the 100th resource element in the frame: {}".format(nr_phy_frame.get_re_frequency_and_time(100)))
    
    # Print the frequency range of the 100th resource element in the frame
    print("Frequency range of the 100th resource element in the frame: {}".format(nr_phy_frame.get_re_frequency_range(100, 10)))
    
    # Print the time range of the 100th resource element in the frame
    print("Time range of the 100th resource element in the frame: {}".format(nr_phy_frame.get_re_time_range(100, 10)))
    
    # Print the resource element index of the 100th subcarrier in the 10th symbol
    print("Resource element index of the 100th subcarrier in the 10th symbol: {}".format(nr_phy_frame.get_re_idx(100, 10 * nr_phy_frame.get_symbol_duration())))
    
    # Print the resource element index of the 100th subcarrier in the 10th slot
    print("Resource element index of the 100th subcarrier in the 10th slot: {}".format(nr_phy_frame.get_re_idx(100, 10 * nr_phy_frame.get_slot_duration())))
    
    # Print the resource element index of the 100th subcarrier in the 10th slot and 10th symbol
    print("Resource element index of the 100th subcarrier in the 10th slot and 10th symbol: {}".format(nr_phy_frame.get_re_idx(100, 10 * nr_phy_frame.get_slot_duration() + 10 * nr_phy_frame.get_symbol_duration())))

if __name__ == "__main__":
    # Run the main function with numerology 1 and 10 slots
    main(1, 10)
    
