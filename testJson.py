import json

with open('materials.json', 'rb') as file:
    data = json.load(file)



# Access the "absorption" object
absorption_data = data["absorption"]

# Access the "rough_concrete" object within "absorption"
rough_concrete_data = absorption_data["Massive constructions and hard surfaces"]["rough_concrete"]

# Retrieve the "coeffs" list
coeffs = rough_concrete_data["coeffs"]

# Print the coefficients
print(coeffs)
