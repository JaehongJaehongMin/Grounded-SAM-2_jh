import json
import sys


with open("object_part.json", "r") as file:
    data = json.load(file)

    # if isinstance(data, dict):
    #     print(f"The data is a dictionary with {len(data)} keys.")
    # elif isinstance(data, list):
    #     print(f"The data is a list with {len(data)} elements.")
    # else:
    #     print("The data is neither a dictionary nor a list.")



for aff in data.keys():
    for obj in data[f"{aff}"]:
        for part in data[f"{aff}"][f"{obj}"]:
            print(f"{part}")

        
