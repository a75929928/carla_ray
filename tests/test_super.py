class Parent:
    def __init__(self):
        self.a = 999
        self.reset()

    def reset(self):
        return NotImplementedError

class Child(Parent):
    # def __init__(self):
    #     super().__init__()

    def reset(self):
        self.a = 1
    

if __name__ == '__main__':
    c = Child()
    print(c.a)
