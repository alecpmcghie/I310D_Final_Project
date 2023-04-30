import tkinter as tk
from PIL import ImageTk, Image
import pickle

# Load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

    
# Create the function to get the inputs and make predictions
def get_inputs():
    age = float(age_entry.get())
    sex = float(sex_entry.get())
    cp = float(cp_entry.get())
    bp = float(bp_entry.get())
    chol = float(chol_entry.get())
    fbs = float(fbs_entry.get())
    ekg = float(ekg_entry.get())
    max_hr = float(max_hr_entry.get())
    exang = float(exang_entry.get())
    st_dep = float(st_dep_entry.get())
    slope = float(slope_entry.get())
    vessels = float(vessels_entry.get())
    thal = float(thal_entry.get())

    inputs = [[age, sex, cp, bp, chol, fbs, ekg, max_hr, exang, st_dep, slope, vessels, thal]]
    prediction = model.predict(inputs)
    print(f"PREDICTION {prediction}")
    if (prediction == 1):
        result_label.config(text="Warning the model predicted POSITIVE, still consult a doctor for proper diagnosis", font=("Arial", 24, "bold"), fg="red", wraplength= 400)
    else:
        result_label.config(text="The model predicted negitive but still consult a doctor for proper diagnosis", font=("Arial", 16), fg="green", wraplength= 400)

# Create the GUI
root = tk.Tk()
root.geometry("600x800")
root.title("MAMA's Heart Disease Prediction")

# creating a left side 
left_frame = tk.Frame(root, bg="light blue", width= 200)

left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

# Add labels and entry boxes for each input variable

label1 = tk.Label(left_frame, text="MAMA's", font=("Arial", 24, "bold"), fg="blue", bg ="light blue")
label1.pack(anchor= "n",padx=(20,0), pady=(20,0))

# create the second label with regular font and black color
label2 = tk.Label(left_frame, text="Healthcare", font=("Arial", 24), fg="black", bg = "light blue")
label2.pack( anchor= "n")

#creating a names section 
names_text = "Marcus Hanania \n Alex Sanders \n Max Diamond \n Alec McGhie"
names_label = tk.Label(left_frame, text= names_text, font=("Arial", 20), fg = "black", bg = "light blue")
names_label.pack(side = tk.BOTTOM, pady= (0,20))


age_label = tk.Label(root, text="Age")
age_label.pack(pady=(20,0))
age_entry = tk.Entry(root)
age_entry.pack()

sex_label = tk.Label(root, text="Sex")
sex_label.pack()
sex_entry = tk.Entry(root)
sex_entry.pack()

cp_label = tk.Label(root, text="Chest pain type")
cp_label.pack()
cp_entry = tk.Entry(root)
cp_entry.pack()

bp_label = tk.Label(root, text="Blood pressure")
bp_label.pack()
bp_entry = tk.Entry(root)
bp_entry.pack()

chol_label = tk.Label(root, text="Cholesterol")
chol_label.pack()
chol_entry = tk.Entry(root)
chol_entry.pack()

fbs_label = tk.Label(root, text="FBS over 120")
fbs_label.pack()
fbs_entry = tk.Entry(root)
fbs_entry.pack()

ekg_label = tk.Label(root, text="EKG results")
ekg_label.pack()
ekg_entry = tk.Entry(root)
ekg_entry.pack()

max_hr_label = tk.Label(root, text="Max heart rate")
max_hr_label.pack()
max_hr_entry = tk.Entry(root)
max_hr_entry.pack()

exang_label = tk.Label(root, text="Exercise angina")
exang_label.pack()
exang_entry = tk.Entry(root)
exang_entry.pack()

st_dep_label = tk.Label(root, text="ST depression")
st_dep_label.pack()
st_dep_entry = tk.Entry(root)
st_dep_entry.pack()

slope_label = tk.Label(root, text="Slope of ST")
slope_label.pack()
slope_entry = tk.Entry(root)
slope_entry.pack()

vessels_label = tk.Label(root, text="Number of vessels fluro")
vessels_label.pack()
vessels_entry = tk.Entry(root)
vessels_entry.pack()

thal_label = tk.Label(root, text="Thallium")
thal_label.pack()
thal_entry = tk.Entry(root)
thal_entry.pack()

# Add a button to submit the inputs and make a prediction
submit_button = tk.Button(root, text="Submit", command=get_inputs)
submit_button.pack()

# Add a label to display the prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
