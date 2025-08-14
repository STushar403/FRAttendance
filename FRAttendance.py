import cv2 as cv
import numpy as np
import time
import os
from insightface.app import FaceAnalysis
import sqlite3
import datetime
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json



#--------------------------------------------------------------------------------------------------------------------#



# Initialize InsightFace

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ROOT = os.path.join(BASE_DIR, "models")

# Initialize InsightFace with correct provider and root path
try:
    app = FaceAnalysis(name='buffalo_s', root=MODEL_ROOT, providers=['CUDAExecutionProvider'])
except:
    app = FaceAnalysis(name='buffalo_s', root=MODEL_ROOT, providers=['CPUExecutionProvider'])

# Prepare The app
app.prepare(ctx_id=0, det_size=(320, 320))



#--------------------------------------------------------------------------------------------------------------------#

#----1----#
def save_embeddings_to_db(db_name, known_faces):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Known_faces_DB (
            name TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')

    for name, embedding in known_faces.items():
        # Save as binary blob
        blob = embedding.tobytes()
        c.execute('INSERT OR REPLACE INTO Known_faces_DB (name, embedding) VALUES (?, ?)', (name, blob))

    conn.commit()
    conn.close()
    return



#----2----#
def load_embeddings_from_db(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS Known_faces_DB (
            name TEXT PRIMARY KEY,
            embedding BLOB
        )
    ''')
    known_faces = {}
    for row in c.execute('SELECT name, embedding FROM Known_faces_DB'):
        name = row[0]
        embedding = np.frombuffer(row[1], dtype=np.float32)
        known_faces[name] = embedding

    conn.close()
    return known_faces



#----3----#
def add_new_embeddings(filename,known_faces,s):
    
    print("[INFO] Loading known faces...")
    
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(s, filename)
        img = cv.imread(img_path)
        faces = app.get(img)
        
        if len(faces) == 0:
            print(f"[WARNING] No face found in {filename}")
            
        else:
            embedding = faces[0].embedding
            #print(embedding)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            name = os.path.splitext(filename)[0]
            known_faces[name] = embedding
            print(f"[INFO] Face loaded: {filename}, embedding norm: {np.linalg.norm(embedding):.2f}")
            return(known_faces)



#----4----#
def delete_extra_data(s,known_faces,db_name):
    
    if known_faces is None:
        return(known_faces)
    
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    temp = []
    known_copy = known_faces.copy()
    for filename in os.listdir(s):
        name = os.path.splitext(filename)[0]
        temp.append(name)
    for x in known_copy:
        if x not in temp:
            c.execute('DELETE FROM Known_faces_DB WHERE name = ?',(x,))
            known_faces.pop(x)
            print (x," is deleted")
    conn.commit()
    conn.close()
    return(known_faces)



#----5----#
def add_column(attend_db,col):
    conn = sqlite3.connect(attend_db)
    c = conn.cursor()
      
    query = f'ALTER TABLE ATTENDENCE ADD COLUMN "{col}" Text'
    c.execute(query)
    conn.commit()
    conn.close()



#----6----#
def add_data(attend_db,col,name,attend_time):
    conn =sqlite3.connect(attend_db)
    c = conn.cursor()
    
    query = f'UPDATE ATTENDENCE SET "{col}" = "{attend_time}" WHERE ID = "{name}"'
    c.execute(query)
    conn.commit()
    conn.close()



#----7----#
def create_attend_db(attend_db,name):
    conn = sqlite3.connect(attend_db)
    c = conn.cursor()
    query = f'CREATE TABLE IF NOT EXISTS ATTENDENCE (ID TEXT PRIMARY KEY)'
    c.execute(query)
    query = f'INSERT INTO ATTENDENCE (ID) VALUES ("{name}")'
    try:
        c.execute(query)
    except:
        pass
    conn.commit()
    conn.close()



#----8----#
def recognize_face(cap,known_faces,attend_db,starting_time,duration):
    while True:
        is_true, frame = cap.read()
        if not is_true:
            print("frame can't found")
            break
        
        faces = app.get(frame)
        
        for face in faces:
            
            bbox = face.bbox.astype(int)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            face_size = (width + height) / 2
            
            if face_size < 100:
                continue
            
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            # Initialize best match
            name = "Unknown"
            min_dist = float("inf")
            
            for known_name, known_embedding in known_faces.items():
                dist = np.linalg.norm(embedding - known_embedding)
                #print(f"[DEBUG] Distance to {known_name}: {dist:.2f}")
                if dist < THRESHOLD and dist < min_dist:
                    min_dist = dist
                    name = known_name
                    if name in recognized_faces:
                        continue
                    recognized_faces.add(name)
                    attend_time = f'{a.hour}:{a.minute}'
                    add_data(attend_db,col,name,attend_time)
                    st = time.time()
                    
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    
                    
                    cv.circle(frame, (center_x, center_y), radius=400, color=(255, 255, 255), thickness=500)
                    cv.putText(frame, f"Welcome {name} ({min_dist:.2f})", (150, 450), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    print(f"{name} Present")
                    while time.time() - st < 3:
                        cv.imshow("InsightFace Recognition", frame)
                        if cv.waitKey(1) & 0xFF == ord('f'):
                            break


        cv.imshow("InsightFace Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('f'):
            break
        
        if time.time()-starting_time >= duration:
            break
    return



#----9----#
def main(selected_path,dur):
    
    
    
    s = selected_path
    if '\\' in s:
        sc = s.split("\\")
        s2 = sc[-1]
    else:
        s2 = s

    duration = dur*60
    db_name = f"{s2}_img.db"
    attend_db = f"{s2}_attendence.db"
    
    
    # Load known faces
    known_faces = load_embeddings_from_db(db_name)

    # Delete extra face data
    known_faces = delete_extra_data(s,known_faces,db_name)


    # Add new face data in face database and names in attendence database

    for filename in os.listdir(s):
            name = os.path.splitext(filename)[0]
            if name not in known_faces:
                add_new_embeddings(filename, known_faces, s)
                create_attend_db(attend_db,name)


    # Save all the face data(Old and New) to database
    
    save_embeddings_to_db(db_name, known_faces)



    # Add new column 

    add_column(attend_db,col)


    # Start webcam
    cap = cv.VideoCapture(0)


    recognize_face(cap,known_faces,attend_db,starting_time,duration)

    #print(box[0],box[1]+250)
    cap.release()
    cv.destroyAllWindows()



def database(selected_path):
    s = selected_path
    s = selected_path
    if '\\' in s:
        sc = s.split("\\")
        s2 = sc[-1]
    else:
        s2 = s
        attend_db = f"{s2}_attendence.db"
    
    # Clear old data and columns
    tree.delete(*tree.get_children())
    tree["columns"] = ()  # reset column list
    
    conn = sqlite3.connect(attend_db)
    cursor = conn.cursor()

    # Get column names
    cursor.execute("PRAGMA table_info(ATTENDENCE)")
    columns = [col[1] for col in cursor.fetchall()]

    tree["columns"] = columns
    tree["show"] = "headings"

    # Set headings dynamically
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)

    # Get and insert all rows
    cursor.execute("SELECT * FROM ATTENDENCE")
    for row in cursor.fetchall():
        tree.insert("", tk.END, values=row)

    conn.close()



#--------------------------------------------------------------------------------------------------------------------#
#GUI Functions



def load_paths():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return []

def save_path(new_path):
    paths = load_paths()
    if new_path not in paths:
        paths.append(new_path)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(paths, f, indent=2)

def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        path_var.set(folder_selected)
        save_path(folder_selected)
        update_dropdown()

def update_dropdown():
    dropdown['menu'].delete(0, 'end')
    for val in load_paths():
        dropdown['menu'].add_command(label=val, command=tk._setit(path_var, val))

def submit():
    selected_path = path_var.get()
    try:
        dur = int(duration_var.get())
        if not os.path.exists(selected_path):
            raise ValueError("Invalid folder path")
        # Use selected_path and dur in your main code
        messagebox.showinfo("Success", f"Path: {selected_path}\nDuration: {dur} minutes")
        root.destroy()  # Close window
        main(selected_path,dur)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def db_view():
    selected_path = path_var.get()
    try:
        if not os.path.exists(selected_path):
            raise ValueError("Invalid folder path")
        # Use selected_path and dur in your main code
        messagebox.showinfo("Success", f"Path: {selected_path}")
        database(selected_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))


#--------------------------------------------------------------------------------------------------------------------#



starting_time = time.time()
recognized_faces = set()
a = datetime.datetime.now()
col = f"{a.day}/{a.month}/{a.year}-{a.hour}:{a.minute}"
# Recognition threshold
THRESHOLD = 0.8
# File to store previous paths
CONFIG_FILE = "paths.json"



#--------------------------------------------------------------------------------------------------------------------#



# GUI Setup
root = tk.Tk()
root.title("Face Recognised Attendance Setup")
root.geometry("600x700")
#root.resizable(False, False)


# Variables
path_var = tk.StringVar()
duration_var = tk.StringVar(value="1")  # default duration
tree = ttk.Treeview(root)
tree.pack(fill=tk.BOTH, expand=True)



#--------------------------------------------------------------------------------------------------------------------#



# Load previous paths
stored_paths = load_paths()
if stored_paths:
    path_var.set(stored_paths[-1])  # Set last used as default

# UI Elements
tk.Label(root, text="Select known_faces Folder:").pack(pady=(10, 5))
initial_value = stored_paths[-1] if stored_paths else "Select a folder"
path_var.set(initial_value)
dropdown = tk.OptionMenu(root, path_var, *([initial_value] + stored_paths))

dropdown.pack(pady=5)

tk.Button(root, text="Browse...", command=browse_folder).pack()

tk.Label(root, text="Duration (minutes):").pack(pady=(15, 5))
tk.Entry(root, textvariable=duration_var).pack()

tk.Button(root, text="Submit", command=submit).pack(pady=20)

btn = tk.Button(root, text="View DataBase", command=db_view)
btn.pack(pady=10)

update_dropdown()
root.mainloop()


#End_By_Tushar