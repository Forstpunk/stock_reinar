# write a machine-readable copy alongside the terminal report
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5 --json report.json


# momentum + gate only (skips fundamentals entirely, much faster)
.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5 --skip-fundamentals


# your own custom watchlist
.venv\Scripts\python.exe -m nse_screener --universe my_list.txt --top 5
my_list.txt (or any universe file) is just one NSE symbol per line, no .NS suffix (e.g. RELIANCE, TCS).





.venv\Scripts\python.exe -m nse_screener --universe universe.txt --top 5