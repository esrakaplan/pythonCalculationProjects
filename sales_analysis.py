import numpy as np
import pandas as pd
from datetime import datetime, timedelta


np.random.seed(42)


dates = pd.date_range(start='2023-01-01', periods=365, freq='D')


sales = pd.DataFrame({
    'Date': dates,
    'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'USB'], 365),
    'Category': np.random.choice(['Electronics', 'Accessories'], 365),
    'Sale_Quantity': np.random.randint(1, 10, 365),
    'Unit_Price': np.random.choice([15, 25, 50, 200, 400], 365),
    'Region': np.random.choice(['Istanbul', 'Ankara', 'Izmir', 'Antalya'], 365),
})


sales['Total_Sale'] = sales['Sale_Quantity'] * sales['Unit_Price']

print(f"Dataset Shape: {sales.shape}")
print(f"Date Range: {sales['Date'].min()} - {sales['Date'].max()}")

print("\nFirst 10 Rows:")
print(sales.head(10))


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



print("\n[TASK 4] Time Series Analysis")
print("-" * 60)


sales['Month'] = sales['Date'].dt.to_period('M')
monthly_sales = sales.groupby('Month')['Total_Sale'].sum()

print("\nMonthly Sales:")
print(monthly_sales.head(12))

print(f"\nHighest Sales Month: {monthly_sales.idxmax()} (${monthly_sales.max():,.2f})")
print(f"Lowest Sales Month: {monthly_sales.idxmin()} (${monthly_sales.min():,.2f})")

# Weekly sales
sales['Week'] = sales['Date'].dt.isocalendar().week
weekly_sales = sales.groupby('Week')['Total_Sale'].mean()

print(f"\nHighest Average Sales Week: {weekly_sales.idxmax()} (${weekly_sales.max():,.2f})")


print("\n[TASK 5] Category Based Analysis")
print("-" * 60)

category_analysis = sales.groupby('Category').agg({
    'Total_Sale': 'sum',
    'Sale_Quantity': 'sum',
    'Unit_Price': 'mean'
}).round(2)

print("\nCategory Summary:")
print(category_analysis)


print("\n[TASK 6] Pivot Table (Product × Region)")
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


print("\n[TASK 7] Data Transformation")
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


print("\n[TASK 8] Data Quality Check")
print("-" * 60)

print(f"\nTotal Rows: {len(sales)}")

print(f"\nMissing Values:\n{sales.isnull().sum()}")

print(f"\nData Types:\n{sales.dtypes}")


print("\n[TASK 9] Correlation Analysis")
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

print("\n" + "="*60)
print("PROJECT 2 COMPLETED")
print("="*60)