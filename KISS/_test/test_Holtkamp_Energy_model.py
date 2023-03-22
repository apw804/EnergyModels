# Test scripts for the Holtkamp Energy Model

#!/usr/bin/env python3




# P_sleep function
# ================
# print(P_sleep(D=1, P_sleep_zero=Macro.P_sleep_zero))
# = 324.0
# WORKING


# eta_PA function
# ===============
#print(eta_PA(eta_PA_max = Macro.eta_PA_max, 
#             gamma = Macro.gamma,
#             P_PA_limit = Macro.P_PA_limit,
#             P_max = Macro.P_max,
#             D = 1))
# = 0.306
# WORKING
# NOTE: Would we use the P_max value across all load levels or substitute this with the current cell P_out value (in watts)?
#
#
# Plot the equation in eta_PA to see how it changes with P_max
# import matplotlib.pyplot as plt
# x = np.linspace(0, 40, 100)
# y = eta_PA(eta_PA_max = Macro.eta_PA_max,
#            gamma = Macro.gamma,
#            P_PA_limit = Macro.P_PA_limit,
#            P_max = x,
#            D = 1)
# plt.plot(x, y)
# plt.show()


# P_PA function
# =============
# print(P_PA(P_max = Macro.P_max,
#             D = 1,
#             eta_PA = eta_PA(eta_PA_max = Macro.eta_PA_max,
#                             gamma = Macro.gamma,
#                             P_PA_limit = Macro.P_PA_limit,
#                             P_max = Macro.P_max,
#                             D = 1),
#             sigma_feed = Macro.sigma_feed
# ))
# = 6.12
# WORKING


# P_RF function
# =============
# print(P_RF(D = 1, W = 10000, P_prime_RF = Macro.P_prime_RF))
# = 12.9
# WORKING


# P_BB function
# =============
# print (P_BB(D = 1, W = 10000, P_prime_BB = Macro.P_prime_BB))
# = 29.4
# WORKING


# P1 function
# ===========
# print(P1(D=1))
# = 62.8780131482834
# This is the P1 value per antenna / radio chain in watts.
# WORKING


# params_P_model function
# =======================
# print(param_P_model(M_sec = Macro.M_sec,
#                     P_1 = P1(D=1, 
#                              W=10000),
#                     delta_p = Macro.delta_p_10MHz,
#                     P_max = Macro.P_max,
#                     chi=1.0))
# = 188.6340394448502 - This is the P_model value per antenna / radio chain 
#                       and per Sector - in watts.
# WORKING