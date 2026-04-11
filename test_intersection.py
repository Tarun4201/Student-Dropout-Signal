import app
df = app.load_and_process_data()
print("Intersection counts:", df['intersection'].value_counts())
