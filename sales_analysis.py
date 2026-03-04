import numpy as np
import pandas as pd
from datetime import datetime, timedelta


np.random.seed(42)


dates = pd.date_range(start='2023-01-01', periods=10, freq='D')


sales = pd.DataFrame({
    'Date': dates,
    'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'USB'], 10),
    'Category': np.random.choice(['Electronics', 'Accessories'], 10),
    'Sale_Quantity': np.random.randint(1, 10, 10),
    'Unit_Price': np.random.choice([15, 25, 50, 200, 400], 10),
    'Region': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Antalya'], 10),
})


sales['Total_Sale'] = sales['Sale_Quantity'] * sales['Unit_Price']

print(f"Dataset Shape: {sales.shape}")
print(f"Date Range: {sales['Date'].min()} - {sales['Date'].max()}")

print("\nSales Data Frame")
print(sales)


# ===== TASK 1: Basic Statistics =====
print("\n[TASK 1] General Statistics")
print("-" * 60)

print(f"\nTotal Sales Amount: ${sales['Total_Sale'].sum():,.2f}")
print(f"Average Sales Amount: ${sales['Total_Sale'].mean():,.2f}")
print(f"Median Sales Amount: ${sales['Total_Sale'].median():,.2f}")
print(f"Standard Deviation: ${sales['Total_Sale'].std():,.2f}")

print(f"\nTotal Transactions: {len(sales)}")
print(f"Total Units Sold: {sales['Sale_Quantity'].sum()}")


print("\n[TASK 2] Product Based Analysis")
print("-" * 60)

product_analysis = sales.groupby('Product').agg({
    'Total_Sale': ['sum', 'mean', 'count'],
    'Sale_Quantity': 'sum'
}).round(2)

product_analysis.columns = [
    'Total_Sales',
    'Average_Sale',
    'Transaction_Count',
    'Total_Quantity'
]

product_analysis = product_analysis.sort_values('Total_Sales', ascending=False)

print("\nProduct Summary:")
print(product_analysis)


print("\n[TASK 3] Region Based Analysis")
print("-" * 60)

region_analysis = sales.groupby('Region')['Total_Sale'].agg(['sum', 'mean', 'count'])

region_analysis.columns = [
    'Total_Sales',
    'Average_Sale',
    'Transaction_Count'
]

region_analysis = region_analysis.sort_values('Total_Sales', ascending=False)

print("\nRegion Summary:")
print(region_analysis)

region_analysis['Percentage'] = (
    region_analysis['Total_Sales'] /
    sales['Total_Sale'].sum() * 100
).round(2)

print("\nRegional Sales Contribution (%):")
print(region_analysis[['Total_Sales', 'Percentage']])


print("\n[TASK 4] Category Based Analysis")
print("-" * 60)

category_analysis = sales.groupby('Category').agg({
    'Total_Sale': 'sum',
    'Sale_Quantity': 'sum',
    'Unit_Price': 'mean'
}).round(2)

print("\nCategory Summary:")
print(category_analysis)


print("\n[TASK 5] Pivot Table (Product × Region)")
print("-" * 60)

pivot_table = pd.pivot_table(
    sales,
    values='Total_Sale',
    index='Product',
    columns='Region',
    aggfunc='sum'
).round(2)

print("\nProduct × Region Sales Matrix:")
print(pivot_table)


print("\n[TASK 6] Data Transformation")
print("-" * 60)

# Add tax and net sales columns
sales['Tax'] = sales['Total_Sale'] * 0.18
sales['Net_Sale'] = sales['Total_Sale'] - sales['Tax']


def sale_level(x):
    if x > 1000:
        return 'High'
    elif x > 300:
        return 'Medium'
    else:
        return 'Low'


sales['Sale_Level'] = sales['Total_Sale'].apply(sale_level)

print("\nAfter Adding Tax and Net Sales:")
print(sales[['Date', 'Product', 'Total_Sale', 'Tax', 'Net_Sale', 'Sale_Level']].head(10))


print("\n[TASK 7] Data Quality Check")
print("-" * 60)

print(f"\nTotal Rows: {len(sales)}")

print(f"\nMissing Values:\n{sales.isnull().sum()}")

print(f"\nData Types:\n{sales.dtypes}")


print("\n[TASK 8] Correlation Analysis")
print("-" * 60)

correlation = sales[['Sale_Quantity', 'Unit_Price', 'Total_Sale']].corr().round(3)

print("\nCorrelation Between Numeric Variables:")
print(correlation)


print("\n[OUTPUT] Save Reports")
print("-" * 60)

# Save main dataset
sales.to_csv('sales_data.csv', index=False, encoding='utf-8')
print("✓ Sales data saved to 'sales_data.csv'")

# Save product analysis
product_analysis.to_csv('product_analysis.csv', encoding='utf-8')
print("✓ Product analysis saved to 'product_analysis.csv'")

# Save pivot table
pivot_table.to_csv('pivot_table.csv', encoding='utf-8')
print("✓ Pivot table saved to 'pivot_table.csv'")
