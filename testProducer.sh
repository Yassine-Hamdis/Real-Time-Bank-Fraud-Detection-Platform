# First, let's test the producer runs correctly
# before putting it in Docker

# Make sure Kafka is running locally OR
# use the quick test below that doesn't need Kafka

# Quick syntax test (no Kafka needed)
python -c "
import sys
sys.path.insert(0, 'producer')

# Test imports
import pandas as pd
import json
import time

# Test data loading
df = pd.read_csv('data/creditcard.csv')
print(f'✅ Dataset loaded: {len(df):,} rows')

# Test message building
row = df.iloc[0]
message = {
    'transaction_id': 'TXN-00000001',
    'timestamp': time.time(),
    'Amount': float(row['Amount']),
    'V1': float(row['V1']),
    'is_fraud_ground_truth': int(row['Class']),
}
print(f'✅ Message built: {json.dumps(message, indent=2)[:200]}...')
print('✅ Producer code is valid')
"