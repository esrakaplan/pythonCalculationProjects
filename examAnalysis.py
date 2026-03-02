import numpy as np
import pandas as pd

print("="*60)
print("PROJECT 1 : Exam Result Analysis")
print("="*60)


np.random.seed(42)
exam_results = np.random.randint(40, 100, size=30)
student_names = [f"Student_{i+1}" for i in range(30)]

print(f"Exam Scores: {exam_results}")


print("\nNumPy Statistics Calculation")
print("-" * 60)

average = np.mean(exam_results)
median = np.median(exam_results)
std_dev = np.std(exam_results)
min_score = np.min(exam_results)
max_score = np.max(exam_results)
variance = np.var(exam_results)

print(f"Average: {average:.2f}")
print(f"Median: {median:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Minimum Score: {min_score}")
print(f"Maximum Score: {max_score}")


print("\n NumPy Boolean Indexing")
print("-" * 60)

successful_student_count = np.sum(exam_results >= 70)
successful_rate = (successful_student_count / len(exam_results)) * 100
successful_scores = exam_results[exam_results >= 70]
failed_scores = exam_results[exam_results < 70]

print(f"70+ scores, student count: {successful_student_count} ({successful_rate:.1f}%)")
print(f"Successful student average scores: {np.mean(successful_scores):.2f}")
print(f"Failed student average scores: {np.mean(failed_scores):.2f}")


print("\nPandas DataFrame")
print("-" * 60)

df = pd.DataFrame({
    'Student': student_names,
    'Score': exam_results,
    'State': ['Sucessfull' if p >= 70 else 'Failed' for p in exam_results],
    'Letter_Grade': ['A' if p >= 90 else 'B' if p >= 80 else 'C' if p >= 70 else 'D' if p >= 60 else 'F' for p in exam_results]
})

print(f"\nDataFrame Info:")
print(df.info())

print(df.head(10))

print("\n Pandas Deep Analysis")
print("-" * 60)

print("\nState Distribution:")
print(df['State'].value_counts())

print("\nLetter Grade Distribution:")
print(df['Letter_Grade'].value_counts().sort_index(ascending=False))

print("\nDescriptive Statistics:")
print(df['Score'].describe())


print("\nFiltering & Sorting")
print("-" * 60)


highest_5 = df.nlargest(5, 'Score')
print("\nEn Highest 5 Scores: ")
print(highest_5)
lowest_5 = df.nsmallest(5, 'Score')
print("\nEn Lowest 5 Scores:")
print(lowest_5)

df['Z_Score'] = (df['Score'] - df['Score'].mean()) / df['Score'].std()
df['Percentage'] = df['Score'].rank(pct=True) * 100

print("\nZ-Score Updated DataFrame:")
print(df.head(10))

State_Sum = df.groupby('State')['Score'].agg(['count', 'mean', 'min', 'max'])
print("\nSummary Based on State:")
print(State_Sum)

Letter_Grade_Sum = df.groupby('Letter_Grade')['Score'].agg(['count', 'mean'])
print("\nSummary Based on Letter Grade :")
print(Letter_Grade_Sum)


df.to_csv('examResults.csv', index=False, encoding='utf-8')
print("✓ Results is saved to 'examResults.csv' file.")

