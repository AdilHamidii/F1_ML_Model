import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1

sessions = []

for round in range(1, 11):
    s = fastf1.get_session(2025, round, 'Q')
    s.load()
    sessions.append(s)

for i in range (len(sessions) -1 ):
    sessions[i].load()


    