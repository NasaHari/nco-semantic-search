import pandas as pd

# Data for the benchmark dataset.
# This is a handcrafted list of queries and their expected output Unit_Codes
# based on common and diverse job roles.
benchmark_data = [
    # --- Queries for Software Developer ---
    {'query': 'Software Developer', 'expected_code': 'U003'},
    {'query': 'person who builds websites and apps', 'expected_code': 'U003'},
    {'query': 'कंप्यूटर प्रोग्राम लिखने वाला', 'expected_code': 'U003'},

    # --- Queries for Pharmacist ---
    {'query': 'Pharmacist', 'expected_code': 'U002'},
    {'query': 'someone who gives out medicine at a drug store', 'expected_code': 'U002'},
    {'query': 'दवा की दुकान पर दवा देने वाला', 'expected_code': 'U002'},

    # --- Queries for Sewing Machine Operator ---
    {'query': 'Sewing Machine Operator', 'expected_code': 'U001'},
    {'query': 'person who stitches clothes in a factory', 'expected_code': 'U001'},
    {'query': 'कपड़े सिलने वाला', 'expected_code': 'U001'},

    # --- Queries for Manager ---
    {'query': 'Manager', 'expected_code': '1'},
    {'query': 'person who directs and evaluates the activities of a company', 'expected_code': '1'},
    {'query': 'कंपनी का प्रबंधन कौन करता है', 'expected_code': '1'},

    # --- Additional Diverse Roles (replace with real codes from your CSV) ---
    {'query': 'Teacher', 'expected_code': '2320.0100'},
    {'query': 'someone who teaches children in a school', 'expected_code': '2320.0100'},
    {'query': 'स्कूल में पढ़ाने वाला', 'expected_code': '2320.0100'},

    {'query': 'Doctor', 'expected_code': '2211.0101'},
    {'query': 'person who treats sick people', 'expected_code': '2211.0101'},
    {'query': 'बीमार लोगों का इलाज करने वाला', 'expected_code': '2211.0101'},

    {'query': 'Farmer', 'expected_code': '6111.0100'},
    {'query': 'person who grows crops in a field', 'expected_code': '6111.0100'},
    {'query': 'खेती करने वाला व्यक्ति', 'expected_code': '6111.0100'},
    
    {'query': 'Accountant', 'expected_code': '2411.0101'},
    {'query': 'manages a company\'s financial records', 'expected_code': '2411.0101'},

    {'query': 'Architect', 'expected_code': '2161.0101'},
    {'query': 'someone who designs buildings', 'expected_code': '2161.0101'},

    {'query': 'Electrician', 'expected_code': '7411.0101'},
    {'query': 'person who fixes electrical wiring', 'expected_code': '7411.0101'},

    {'query': 'Plumber', 'expected_code': '7126.0201'},
    {'query': 'someone who repairs pipes and toilets', 'expected_code': '7126.0201'},
]

# Create a pandas DataFrame
benchmark_df = pd.DataFrame(benchmark_data)

# Save the DataFrame to a CSV file
benchmark_df.to_csv('benchmark.csv', index=False)

print("Benchmark dataset created successfully!")