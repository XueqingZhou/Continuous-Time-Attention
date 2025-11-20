#!/bin/bash

# Script to download and prepare tokenizers

echo "Preparing BERT tokenizer..."

python << EOF
from transformers import BertTokenizer
import os

# Create tokenizer directory
os.makedirs('./tokenizer', exist_ok=True)

# Download and save BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.save_pretrained('./tokenizer/bert-base-uncased')

print("✓ BERT tokenizer saved to ./tokenizer/bert-base-uncased")

EOF

echo "Done! Tokenizer is ready."

