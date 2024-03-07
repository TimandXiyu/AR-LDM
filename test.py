import torch

gears = [11, 12, 13, 14, 15, 17, 19, 21, 24, 27, 30]
for gear in gears:
    speed = 90 * (50 / gear) * 2 / 60 * 3.6
    print(f"{gear} {speed:.2f}")