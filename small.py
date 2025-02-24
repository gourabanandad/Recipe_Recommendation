import pandas as pd

df = pd.read_csv("C:\\Users\\goura\\Desktop\\Recipe Recommendation System\\dataset\\full_dataset.csv")

# Keep only the top 50,000 recipes (or any smaller number)
df_small = df.sample(n=100000, random_state=42)  

df_small.to_csv("small_dataset.csv", index=False)