import pandas as pd

folder = "results/dt/"
file = "dt_results.csv"

df = pd.read_csv(folder+file)
# print(max(df['mean_train_score']))
# print(min(df['std_train_score']))
# print(max(df['mean_test_score']))
# print(min(df['std_test_score']))
# print(max(df['holdout_test_score']))

columns = [
	'data',
	'scaled',
	'pruned',
	'inner_nodes_prev',
	'inner_nodes_post',
	'mean_train_score',
	'std_train_score',
	'mean_test_score',
	'std_test_score',
	'holdout_test_score'
]

df[columns].to_csv(folder+"formated_"+file, float_format='%.4f', index=False)