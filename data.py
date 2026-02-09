import csv
import random

# Define sectors and states for the Zimbabwe context
sectors = ['Formal', 'Informal']
borrower_types = {
    'Formal': ['Civil Servant', 'Private Sector Employee', 'SME-Manufacturing', 'SME-Agro-Processing'],
    'Informal': ['Cross-Border Trader', 'Market Vendor', 'Tuckshop Owner', 'Artisan-Miner', 'Street Vendor']
}
states = ['S1', 'S2', 'S3', 'S4']

# Create the CSV file
with open('dsmco_test_data_1000.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['loan_id', 'sector', 'borrower_type', 'initial_state', 'final_state', 'amount_usd'])
    
    for i in range(1, 1001):
        sector = random.choice(sectors)
        borrower_type = random.choice(borrower_types[sector])
        initial_state = random.choice(states)
        # Randomly assign a final state to simulate a transition
        final_state = random.choice(states)
        # Randomize amount (USD) based on typical loan sizes in the Zimbabwe context
        amount = random.randint(100, 5000) if sector == 'Formal' else random.randint(50, 1000)
        
        writer.writerow([f'L{i:03}', sector, borrower_type, initial_state, final_state, amount])

print("1000-line CSV file 'dsmco_test_data_1000.csv' has been created.")