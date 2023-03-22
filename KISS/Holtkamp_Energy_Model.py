# A 'simple' script to reproduce the Holtkamp et al. (2013) results.
# DOI: 10.1109/LCOMM.2013.091213.131042

# P_supply = supply power consumption of a BS
# P0 = load-independent power consumption
# P1 = maximum supply power consumption
# delta_p = linear gradient of power consumption
# P_max = maximum total transmission power (watts)
# M_sec = number of sectors
# P_sleep = power consumption in sleep mode

# W = transmission bandwidth (Hz)
# D = number of BS radio chains/antennas

# P_PA = power amplifier power consumption
# P_RF = RF power consumption
# P_BB = baseband processing power consumption

# sigma_feed = feedline loss (dB)
# sigma_DC = DC power loss (dB)
# sigma_MS = Mains Supply power loss (dB)
# sigma_cool = cooling power loss (dB)

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Macro:
    """Macro cell parameters."""
    P_PA_limit:     float =  80.0    # watts
    eta_PA_max:     float =   0.36
    gamma:          float =   0.15
    P_prime_BB:     float =  29.4    # watts
    P_prime_RF:     float =  12.9    # watts
    sigma_feed:     float =   0.50
    sigma_DC:       float =   0.075
    sigma_cool:     float =   0.100
    sigma_MS:       float =   0.075
    M_sec:            int =   3
    P_max:          float =  40.0    # watts
    P1:             float = 460.0    # watts
    delta_p_10MHz:  float =   4.2
    P_sleep_zero:   float = 324.0    # watts


def P_sleep(D, P_sleep_zero):
    """
    Returns the power consumption of the base station in sleep mode.

    Parameters
    ----------
    model : str
        The model to use for the calculation.
    D : int
        Number of BS radio chains/antennas
    P_sleep_zero : float
        Power consumption of the base station in sleep mode (watts). 
        This is a "reference value for the single antenna BS chosen such that 
        P_sleep matches the complex model value for two antennas" 
        (Holtkamp et al. 2013).
    """



    return D * P_sleep_zero


def eta_PA(eta_PA_max, gamma, P_PA_limit, P_max, D):
    """
    Returns the power amplifier efficiency.
    """

    eta_PA = eta_PA_max * (1 - gamma * np.log2 (P_PA_limit / (P_max / D)))
    return eta_PA


def P_PA(P_max: float, D: int, eta_PA: float, sigma_feed: float):
    """
    Returns the power amplifier power consumption of the base station in watts.
    """
    return P_max / D * eta_PA * (1 - sigma_feed)


def P_RF(D: int = 1, W: int = 10e3, P_prime_RF: float = Macro.P_prime_RF):
    """
    Returns the RF power consumption of the base station in watts.

    Parameters
    ----------
    D : int
        Number of BS radio chains/antennas
    W : int
        Transmission bandwidth (Hz)
    P_prime_RF : float
        RF power consumption (watts)
      
    Returns
    -------
    float
        RF power consumption of the base station in watts.
    """

    return  D * (W / 10e3) * P_prime_RF


def P_BB(D: int = 1, W: int = 10e3, P_prime_BB: float = Macro.P_prime_BB):
    """
    Returns the baseband processing power consumption of the base station 
    in watts.
    """
    return  D * (W / 10e3) * P_prime_BB


def param_P_1(P_0: float, delta_P: float = Macro.delta_p_10MHz, 
                P_max: float = Macro.P_max):
    """
    Returns the parameterised model maximum supply power consumption of the base
    station in watts.
    """
    return P_0 + delta_P * P_max

print(param_P_1(P_0=P_sleep(model='Holtkamp', D=1, P_sleep_zero=Macro.P_sleep_zero)
                delta_P=Macro.delta_p_10MHz, 
                P_max=Macro.P_max))


def complex_P_1(D: int, W: int = 10e3):
    """
    Returns the complex model maximum supply power consumption of the 
    base station in watts.

    Parameters
    ----------
    """

    BB = P_BB(D, W, P_prime_BB = Macro.P_prime_BB)
    RF = P_RF(D, W, P_prime_RF = Macro.P_prime_RF)
    PA = P_PA(P_max = Macro.P_max,
              D=D,
              eta_PA = eta_PA(eta_PA_max = Macro.eta_PA_max,
                              gamma = Macro.gamma,
                              P_PA_limit = Macro.P_PA_limit,
                              P_max = Macro.P_max,
                              D=D),
              sigma_feed = Macro.sigma_feed)
    
    o_DC   = Macro.sigma_DC
    o_MS   = Macro.sigma_MS
    o_cool = Macro.sigma_cool

    numerator = BB + RF + PA
    denominator = (1 - o_DC) * (1 - o_MS) * (1 - o_cool) 

    return numerator / denominator





def P_supply(x, D=1, complex: bool = False):
    """
    Returns the supply power consumption of the base station at a given 
    load `x`.

    Parameters
    ----------
    x : float
        Load of the base station (which is the fractional use of time and frequency resources [dt * df_c])

    D : int, optional
        Number of BS radio chains/antennas, by default 1
    complex : bool, optional
        Whether to use the complex or simple power consumption model, by default False

    Returns
    -------
    float
        Supply power consumption of the base station.
    """

    # Compute the load-independent power consumption
    P0 = P_sleep(D=1, P_sleep_zero=Macro.P_sleep_zero)


    if complex:
      if 0 < x <= 1:
          return Macro.M_sec * complex_P1(D=D, W=10000) + Macro.delta_p_10MHz * Macro.P_max * (x - 1)
      else:
          return Macro.M_sec * P0
    else:
      if 0 < x <= 1:
          return Macro.M_sec * simple_P1(P0=P0) + Macro.delta_p_10MHz * Macro.P_max * (x - 1)
      else:
          return Macro.M_sec * P0





#if __name__ == "__main__":
    # Create a list of loads
    loads = np.arange(0, 1.01, 0.01).astype(float)

    # Run the main function for each load returning the simple power consumption
    # Create a list to hold the results
    simple_model = []
    for load in loads:
        print(f"[SIMPLE] Load: {load:.2f}, Power cons.:{P_supply(x=load, complex=False):.4f} W")
        # Append the load and result to the list
        simple_model.append((load, P_supply(x=load, complex=False)))


    # Run the main function for each load returning the complex power consumption
    # Create a list to hold the results
    complex_model = []
    for load in loads:
        print(f"[COMPLEX] Load: {load:.2f}, Power cons.:{P_supply(x=load, complex=True):.4f} W")
        # Append the load and result to the list
        complex_model.append((load, P_supply(x=load, complex=True)))
    
    # Plot the results
    import matplotlib.pyplot as plt
    plt.plot(*zip(*simple_model), label="Simple model")
    plt.plot(*zip(*complex_model), label="Complex model")
    plt.xlabel("Load")
    plt.ylabel("Power consumption (W)")
    plt.legend()
    # Shift the x-axis and y-axis  to the origin
    plt.gca().spines['left'].set_position('zero')
    plt.gca().spines['bottom'].set_position('zero')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Show the plot
    plt.show()
