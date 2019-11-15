import time

def return_data():
    i = 0
    while True:
        i = i+1
        yield (i)
        time.sleep(0.5)