"""Feature engineering helpers for the income prediction dataset.

Keep transformations additive and deterministic so tests remain stable.
"""
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add lightweight derived features without removing originals.

	Adds:
	- `capital_gain_binary`: 1 if `capital-gain` > 0 else 0
	- `age_bucket`: small categorical bucket of age
	"""
	df = df.copy()

	# capital gain binary indicator
	if 'capital-gain' in df.columns:
		df['capital_gain_binary'] = (df['capital-gain'] > 0).astype(int)

	# age buckets (deterministic bins)
	if 'age' in df.columns:
		bins = [0, 25, 35, 45, 55, 65, 100]
		labels = ['<=25', '26-35', '36-45', '46-55', '56-65', '66+']
		df['age_bucket'] = pd.cut(df['age'], bins=bins, labels=labels, right=True, include_lowest=True)
		# Fill any NA buckets (shouldn't normally happen) with a string
		df['age_bucket'] = df['age_bucket'].astype(str)

	return df
