import time
import csv
from langchain_openai import ChatOpenAI

# 1. Setup the Model
print("🔌 Connecting to Dr. Llama...")
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="medical-llama-3-8b",
    temperature=0,
)

# 2. Define Test Questions
test_cases = [
    "What are the symptoms of Type 2 Diabetes?",
    "Explain the treatment for hypertension.",
    "What is the difference between viral and bacterial infection?",
    "List three side effects of Ibuprofen.",
    "How do you diagnose asthma in children?"
]

# 3. Run Benchmark
results = []
print("🚀 Starting Benchmark on RTX 3050 Ti...")

for i, question in enumerate(test_cases):
    print(f"   Testing Q{i+1}...")
    
    start_time = time.time()
    response = llm.invoke(question) # Run inference
    end_time = time.time()
    
    duration = end_time - start_time
    tokens = len(response.content.split()) # Rough estimate
    tps = tokens / duration # Tokens Per Second
    
    results.append([question, f"{duration:.2f}s", tokens, f"{tps:.2f}"])

# 4. Save Results
csv_filename = "performance_metrics.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Question", "Latency (Seconds)", "Tokens Generated", "Tokens/Sec (TPS)"])
    writer.writerows(results)

print(f"\n✅ Benchmark Complete! Data saved to {csv_filename}")