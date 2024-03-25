class Parent:
    def __init__(self):
        self.a = 1
    def get_data(self, data):
        print(data)

class Child(Parent):
    def __init__(self):
        super().__init__()
        # self.b = 2
    def get_data(self, data):
        super().get_data(data)
        # print(self.b)

if __name__ == '__main__':
    c = Child()
    c.get_data(1)
