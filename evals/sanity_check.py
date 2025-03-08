import pandas as pd

ref_df = pd.read_csv("/lustre/fs01/portfolios/dir/users/jeromek/savanna-test/evals/needle_in_a_haystack/checks/7b-256k/mp8cp1/ref.csv")
test_df = pd.read_csv("/lustre/fs01/portfolios/dir/users/jeromek/savanna-test/evals/needle_in_a_haystack/checks/7b-256k/mp8cp1/2048_100.csv", index_col=0)
random_df = pd.read_csv("/lustre/fs01/portfolios/dir/users/jeromek/savanna-test/evals/needle_in_a_haystack/7b-256k-random/2048_100.csv")

def get_diff(df1, df2):
    diff = df1 - df2
    assert diff['haystack_length'].sum() == 0
    assert diff['depth'].sum() == 0

    diff['haystack_length'] = df1['haystack_length']
    diff['depth'] = df1['depth']
    diff.set_index(['haystack_length', 'depth'], inplace=True)
    return diff

test_ref_diff = get_diff(test_df, ref_df)

summary_test_diff = test_ref_diff.groupby(['haystack_length'])['score'].mean()
print(f"Test vs Ref")
print(summary_test_diff)

test_random_diff = get_diff(test_df, random_df)
summary_test_random_diff = test_random_diff.groupby(['haystack_length'])['score'].mean()
print(f"Test vs Random")
print(summary_test_random_diff)

random_ref_diff = get_diff(random_df, ref_df)
summary_random_ref_diff = random_ref_diff.groupby(['haystack_length'])['score'].mean()
print(f"Random vs Ref")
print(summary_random_ref_diff)
