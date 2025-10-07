"""Test script to verify Bedrock connection and diagnose issues"""
import sys
sys.path.append('./backend')

from config import config
from ai_generator import AIGenerator

print("Testing Bedrock connection...")
print(f"AWS Region: {config.AWS_REGION}")
print(f"Model ID: {config.BEDROCK_MODEL_ID}")
print(f"Access Key ID: {config.AWS_ACCESS_KEY_ID[:10]}..." if config.AWS_ACCESS_KEY_ID else "Not set")

try:
    # Initialize AI Generator
    ai_gen = AIGenerator(
        config.AWS_ACCESS_KEY_ID,
        config.AWS_SECRET_ACCESS_KEY,
        config.AWS_REGION,
        config.BEDROCK_MODEL_ID
    )
    print("✓ AIGenerator initialized successfully")

    # Test a simple query
    print("\nTesting simple query...")
    response = ai_gen.generate_response("What is 2+2?")
    print(f"Response: {response}")
    print("✓ Query successful!")

except Exception as e:
    print(f"✗ Error: {type(e).__name__}")
    print(f"Message: {str(e)}")
    import traceback
    traceback.print_exc()
