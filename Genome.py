class Genome:
    def __init__(self, selection: list[int], fit=0, sets_len=2):
        self.selection: list[int] = selection
        self.fit= fit

    def __repr__(self):
        return f"fit: {self.fit} selection : {self.selection}"