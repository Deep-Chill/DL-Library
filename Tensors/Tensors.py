class Tensor:

    def __init__(self, vector):
        if type(vector) == list:
            self.Tensor = vector
        else:
            print("Invalid input!")
    def __str__(self):
        return f"Your Tensor is: {self.Tensor}"


a = Tensor([1])
print(a)