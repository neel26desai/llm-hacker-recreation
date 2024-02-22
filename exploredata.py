import datasets
ds =  datasets.load_dataset('knowrohit07/know_sql')
print(ds)
print(ds['validation'][3])