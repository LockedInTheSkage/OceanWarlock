import csv

class CSVReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data