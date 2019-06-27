class City(str):
    def __hash__(self):
        return ord(self[0])


# Create a dict and assign
data = {
    City("Rome"): 4,
    City("San Francisco"): 3,
    City("New York"): 5,
    City("Barcelona"): 2,
}

print("Done")