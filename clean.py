import pandas as pd
import re

def clean_telecom_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, skiprows=6, header=0)

    # Remove the first 6 rows
    #df = df.iloc[6:]
    # Remove half-year columns (1H 20XX, 2H 20XX patterns)
    half_year_cols = [col for col in df.columns 
                     if re.match(r'^[12]H\s20\d{2}$', str(col))]
    df = df.drop(columns=half_year_cols)

    # 1. Remove empty columns (columns with all NaN values)
    df = df.dropna(axis=1, how='all')

    # 2. Clean column names (remove extra spaces and special characters)
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)  # Remove special characters

    # 3. Clean numeric columns stored as strings
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to clean and convert to numeric
            df[col] = df[col].str.replace(' ', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # 4. Handle missing values
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df.loc[:, categorical_cols] = df[categorical_cols].fillna('Unknown')

    # 5. Standardize operator names (remove extra spaces)
    if 'Operator name' in df.columns:
        df['Operator name'] = df['Operator name'].str.strip()

    # 6. Standardize country names (remove extra spaces and special characters)
    if 'Country' in df.columns:
        df['Country'] = df['Country'].str.strip()
        df['Country'] = df['Country'].str.replace(r'[^\w\s]', '', regex=True)

    # 7. Remove duplicate rows
    df = df.drop_duplicates()

    # 8. Reset index after cleaning
    df = df.reset_index(drop=True)

    return df

# Usage example:
if __name__ == "__main__":
    cleaned_data = clean_telecom_data('TAA1.csv')
    cleaned_data.to_csv('cleaner.csv', index=False)