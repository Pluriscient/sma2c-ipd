import axelrod as axl
from axelrod.strategies import Cooperator, Defector, TitForTat, ForgivingTitForTat
from axelrod_agent import SMA2CPlayer
import json

# players = [Cooperator, Defector, TitForTat, ForgivingTitForTat, axl.]
# players = (Cooperator(), Defector(), SMA2CPlayer('outputs/1592207053-sma2c-70000'))

# tournament = axl.Tournament(players)
# results = tournament.play()
# print(results)
axl.seed(10)
players = [s() for s in axl.strategies if s in axl.axelrod_first_strategies]
players.append(SMA2CPlayer('outputs/1592601088-sma2c-noseed-10000-smaller-mid-half'))
game = axl.Game(r=2, s=-1, t=3, p=0)
t = axl.Tournament(players, turns=25, repetitions=100, game=game)
res = t.play(filename='tournament-results.csv')
print(res.ranked_names)
res.write_summary('summary-3.csv')
p = axl.Plot(res)