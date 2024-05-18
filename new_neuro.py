with open('car.txt', 'r') as file:
    data = file.readlines()[1:]


table = [row.split() for row in data]


years = [int(row[1]) for row in table if row[1] != 'None']
max_year = max(years)
mean_year = sum(years) / len(years)

km = [int(row[2]) for row in table if row[2] != 'None']
max_km = max(km)
mean_km = sum(km) / len(km)

price = [int(row[0]) for row in table if row[0] != 'None']
max_price = max(price)
mean_price = sum(price) / len(price)


for row in table:
    row[0] = int(row[0]) / max_price
    row[1] = int(row[1]) / max_year if row[1] != 'None' else mean_year / max_year
    row[2] = int(row[2]) / max_km if row[2] != 'None' else mean_km / max_km


out = 'car_step2.txt'


with open(out, 'w') as output_file:
    output_file.write("price_eur\tyear\tkm_age\tmark\n")

    for row in table:
        output_file.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

print(f"Normalized table saved to '{out}' successfully.")
