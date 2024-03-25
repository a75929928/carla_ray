from time import time, sleep
class CustomTimer:

    def __init__(self):
        self.start_time = time()

    def past_time(self):
        return time() - self.start_time
    
if __name__ == '__main__':
    a = CustomTimer()
    sleep(1)
    b = CustomTimer()
    sleep(2)
    print(a.past_time())
    print(b.past_time())