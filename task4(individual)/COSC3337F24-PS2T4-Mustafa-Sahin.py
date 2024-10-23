import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('task4(individual)/data/HW2021.csv')

# Normalize continuous columns
scaler = MinMaxScaler()
df[['min_temp', 'rainfall', 'windspeed', 'humidity']] = scaler.fit_transform(df[['min_temp', 'rainfall', 'windspeed', 'humidity']])

# Assign ordinal values to cloud cover
cloud_mapping = {
    'Fair': 1, 'Fair / Windy': 2, 'Partly Cloudy': 3, 'Partly Cloudy / Windy': 4, 
    'Mostly Cloudy': 5, 'Mostly Cloudy / Windy': 6, 'Cloudy': 7, 'Cloudy / Windy': 8, 
    'Fog': 9, 'Haze': 10, 'Light Rain': 11, 'Light Rain with Thunder': 12, 'Thunder': 13, 
    'Thunder / Windy': 14, 'Heavy T-Storm': 15, 'Thunder in the Vicinity': 16, 'T-Storm': 17
}
df['cloudcover'] = df['cloudcover'].map(cloud_mapping)

# Fill missing cloudcover values with the most frequent value (mode)
most_frequent_cloudcover = df['cloudcover'].mode()[0]
df['cloudcover'].fillna(most_frequent_cloudcover, inplace=True)

# Compute distance between days (Euclidean for continuous + categorical difference for cloud cover)
def calculate_distance(day1, day2, w1=1, w2=1):
    try:
        cloud_distance = abs(day1['cloudcover'] - day2['cloudcover'])
        euclidean_distance = np.sqrt(np.sum((day1[['min_temp', 'rainfall', 'windspeed', 'humidity']] - day2[['min_temp', 'rainfall', 'windspeed', 'humidity']])**2))
        return w1 * cloud_distance + w2 * euclidean_distance
    except Exception as e:
        print(f"Error calculating distance: {e}")
        return np.nan

# Calculate outlier scores (OLS)
outlier_scores = []
for i in range(len(df)):
    day_distances = [calculate_distance(df.iloc[i], df.iloc[j]) for j in range(len(df)) if i != j]
    
    # Only append the mean if there are valid distances
    if len(day_distances) > 0 and not all(np.isnan(day_distances)):
        outlier_scores.append(np.nanmean(day_distances))
    else:
        outlier_scores.append(np.nan)

df['OLS'] = outlier_scores

# Check for NaN values in the OLS
print(f"NaN values in OLS: {df['OLS'].isna().sum()}")

# Save augmented dataset with OLS
df.to_csv('task4(individual)/output/HW2021_augmented.csv', index=False)

# Print top 4 and bottom 2 outliers
sorted_df = df.sort_values(by='OLS', ascending=False)
print("Top 4 Outliers:\n", sorted_df.head(4))
print("\nMost Normal Days:\n", sorted_df.tail(2))
