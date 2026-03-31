import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Name": ["A", "B", "C", "D", "E"],
    "Marks": [70, 85, 90, 60, 75]
}

df = pd.DataFrame(data)

print("\n--- Pandas ---")
print("Average Marks:", df["Marks"].mean())
print("Maximum Marks:", df["Marks"].max())
print("Minimum Marks:", df["Marks"].min())

print("\n--- Plotting Graph ---")
plt.bar(df["Name"], df["Marks"])
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Student Marks Chart")
plt.show()