# 5G; NR; Physical layer procedures for data (3GPP TS 38.214 version 17.4.0 Release 17)


# 5.1.3 Modulation order, target code rate, redundancy version and transport block size determination
"""
Basic method:
-------------
1) Defaults to [Table 5.1.3.1-4: MCS index table 4 for PDSCH]
2) Input: MCS value (int)
3) Return: Spectral Efficiency (bps/Hz)

Advanced method:
----------------
** Not yet implemented, but should look like this: **
1) Maps the DCI (Downlink Control Indicator) format to the correct MCS table
2) Takes the MCS value and finds corresponding:
    - modulation order (Q_m)
    - target code rate (R)
3) Pass these parameters to the Transport Block Size (TBS) determining function
"""

import json

def save_table_as_json(table, filename):
    with open(filename, 'w') as file:
        json.dump(table, file, indent=4) 

mcs_table_1 = {
  0:  (2, 120, 0.2344),
  1:  (2, 157, 0.3066),
  2:  (2, 193, 0.3770),
  3:  (2, 251, 0.4902),
  4:  (2, 308, 0.6016),
  5:  (2, 379, 0.7402),
  6:  (2, 449, 0.8770),
  7:  (2, 526, 1.0273),
  8:  (2, 602, 1.1758),
  9:  (2, 679, 1.3262),
  10: (4, 340, 1.3281),
  11: (4, 378, 1.4766),
  12: (4, 434, 1.6953),
  13: (4, 490, 1.9141),
  14: (4, 553, 2.1602),
  15: (4, 616, 2.4063),
  16: (4, 658, 2.5703),
  17: (6, 438, 2.5664),
  18: (6, 466, 2.7305),
  19: (6, 517, 3.0293),
  20: (6, 567, 3.3223),
  21: (6, 616, 3.6094),
  22: (6, 666, 3.9023),
  23: (6, 719, 4.2129),
  24: (6, 772, 4.5234),
  25: (6, 822, 4.8164),
  26: (6, 873, 5.1152),
  27: (6, 910, 5.3320),
  28: (6, 948, 5.5547),
  29: (2, 'reserved', 'reserved'),
  30: (4, 'reserved', 'reserved'),
  31: (6, 'reserved', 'reserved')}

mcs_table_2 = {
  0:  (2, 120,   0.2344),
  1:  (2, 193,   0.3770),
  2:  (2, 308,   0.6016),
  3:  (2, 449,   0.8770),
  4:  (2, 602,   1.1758),
  5:  (4, 378,   1.4766),
  6:  (4, 434,   1.6953),
  7:  (4, 490,   1.9141),
  8:  (4, 553,   2.1602),
  9:  (4, 616,   2.4063),
  10: (4, 658,   2.5703),
  11: (6, 466,   2.7305),
  12: (6, 517,   3.0293),
  13: (6, 567,   3.3223),
  14: (6, 616,   3.6094),
  15: (6, 666,   3.9023),
  16: (6, 719,   4.2129),
  17: (6, 772,   4.5234),
  18: (6, 822,   4.8164),
  19: (6, 873,   5.1152),
  20: (8, 682.5, 5.3320),
  21: (8, 711,   5.5547),
  22: (8, 754,   5.8906),
  23: (8, 797,   6.2266),
  24: (8, 841,   6.5703),
  25: (8, 885,   6.9141),
  26: (8, 916.5, 7.1602),
  27: (8, 948,   7.4063),
  28: (2, 'reserved', 'reserved'),
  29: (4, 'reserved', 'reserved'),
  30: (6, 'reserved', 'reserved'),
  31: (8, 'reserved', 'reserved')}

mcs_table_3 = {
 0:  (2, 30,  0.0586),
 1:  (2, 40,  0.0781),
 2:  (2, 50,  0.0977),
 3:  (2, 64,  0.1250),
 4:  (2, 78,  0.1523),
 5:  (2, 99,  0.1934),
 6:  (2, 120, 0.2344),
 7:  (2, 157, 0.3066),
 8:  (2, 193, 0.3770),
 9:  (2, 251, 0.4902),
 10: (2, 308, 0.6016),
 11: (2, 379, 0.7402),
 12: (2, 449, 0.8770),
 13: (2, 526, 1.0273),
 14: (2, 602, 1.1758),
 15: (4, 340, 1.3281),
 16: (4, 378, 1.4766),
 17: (4, 434, 1.6953),
 18: (4, 490, 1.9141),
 19: (4, 553, 2.1602),
 20: (4, 616, 2.4063),
 21: (6, 438, 2.5664),
 22: (6, 466, 2.7305),
 23: (6, 517, 3.0293),
 24: (6, 567, 3.3223),
 25: (6, 616, 3.6094),
 26: (6, 666, 3.9023),
 27: (6, 719, 4.2129),
 28: (6, 772, 4.5234),
 29: (2, 'reserved', 'reserved'),
 30: (4, 'reserved', 'reserved'),
 31: (6, 'reserved', 'reserved')}

mcs_table_4 = {
 0:  (2,  120,    0.2344),
 1:  (2,  193,    0.3770),
 2:  (2,  449,    0.8770),
 3:  (4,  378,    1.4766),
 4:  (4,  490,    1.9141),
 5:  (4,  616,    2.4063),
 6:  (6,  466,    2.7305),
 7:  (6,  517,    3.0293),
 8:  (6,  567,    3.3223),
 9:  (6,  616,    3.6094),
 10: (6,  666,    3.9023),
 11: (6,  719,    4.2129),
 12: (6,  772,    4.5234),
 13: (6,  822,    4.8164),
 14: (6,  873,    5.1152),
 15: (8,  682.5,  5.3320),
 16: (8,  711,    5.5547),
 17: (8,  754,    5.8906),
 18: (8,  797,    6.2266),
 19: (8,  841,    6.5703),
 20: (8,  885,    6.9141),
 21: (8,  916.5,  7.1602),
 22: (8,  948,    7.4063),
 23: (10, 805.5,  7.8662),
 24: (10, 853,    8.3301),
 25: (10, 900.5,  8.7939),
 26: (10, 948,    9.2578),
 27: (2,  'reserved', 'reserved'),
 28: (4,  'reserved', 'reserved'),
 29: (6,  'reserved', 'reserved'),
 30: (8,  'reserved', 'reserved'),
 31: (10, 'reserved', 'reserved')}

# 4.1 Power allocation for downlink

