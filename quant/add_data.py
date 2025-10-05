import pandas as pd

def add_line(file_name: str):
    columns = ['Date', 'Close', 'Volume']
    df = pd.read_csv(file_name)

    new_row = {}
    for col in columns:
        value = input(f"Enter value for {col}: ")
        # For 'Close' and 'Volume' you can convert to float/int if you want
        if col == 'Close':
            value = float(value)
        elif col == 'Volume':
            value = int(value)
        # For 'Date' keep as string
        new_row[col] = value

    # Add the row
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file_name, index=False)
    print("Row added!")

add_line('^SPX_historical_data_1d.csv')


