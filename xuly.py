import pandas as pd

df = pd.read_csv("data2.csv")

duplicate_interactions = df[df.duplicated(subset=['user_id', 'title'], keep=False)]

duplicate_interactions.sort_values(by=['user_id', 'title'])

df.drop_duplicates(subset=['user_id', 'title'], keep='first', inplace=True)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s') + pd.Timedelta(hours=7)

df = df.sort_values(["user_id", "timestamp"])

df.to_csv("clean_data.csv", index=False, encoding='utf-8-sig')