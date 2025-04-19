# Face-Recognition-app-using-media-pip
First face is detected ancoded and saved with some user info and then recognized
import cv2
import numpy as np
from PIL import Image, ImageTk
import sqlite3
import tkinter as tk
from tkinter import messagebox, Toplevel
from tkcalendar import Calendar
import mediapipe as mp
from datetime import datetime 
from datetime import date
import pandas as pd
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
fac_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

running = True
num_faces = 0
image = None
input_window = None
main_window = None
video = None
def download():
    conn = sqlite3.connect('facereg.db')
    query = "SELECT * FROM attendance"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    file_path = r'C:\Users\HP\Downloads\attendance_data.xlsx'
    
    df.to_excel(file_path, index=False)
    
    messagebox.showinfo("Success", f"Attendance data successfully downloaded to {file_path}")
def fetch_students():
    conn = sqlite3.connect("facereg.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM registration")
    rows = cursor.fetchall()
    conn.close()
    return rows

def fetch_dates():
    conn = sqlite3.connect("facereg.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(attendance)")
    columns = cursor.fetchall()
    conn.close()
    return [col[1] for col in columns if col[1] not in ('id', 'name')]

def add_date_column(date):
    conn = sqlite3.connect('facereg.db')
    cursor = conn.cursor()
    try:
        cursor.execute(f'ALTER TABLE attendance ADD COLUMN "{date}" TEXT DEFAULT "Absent"')
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, no action needed
        pass
    conn.close()

def check_duplicates(id, encode, name, age, subject, registration_time):
    data = fetch_students()
    i = j = k = 0
    for row in data:
        if row[0] == id:
            i = 1
        if row[1] == name:
            j = 1
    if i == 1 and j == 1:
        messagebox.showwarning("Warning", "User with same id and name already registered")
        input_window.destroy()
    elif i == 1 and j == 0:
        messagebox.showwarning("Warning", "User with same id already registered")
        input_window.destroy()
    
    elif i == 0 and j ==1:
        messagebox.showwarning("Warning", "User with same name already registered")
        input_window.destroy()
    else:
        save(id, encode, name, age, subject, registration_time)
        return

def add_attendance(id, name):
    conn = sqlite3.connect('facereg.db')
    cursor = conn.cursor()
    
    attendance_date = date.today().strftime("%Y-%m-%d")
    
    existing_dates = fetch_dates()
    if attendance_date not in existing_dates:
        add_date_column(attendance_date)
    
    cursor.execute(f'''
    INSERT OR REPLACE INTO attendance (id, name, "{attendance_date}")
    VALUES (?, ?, "Absent")
    ''', (id, name))
    
    conn.commit()
    conn.close()

def save_attendance(id, name, date):
    conn = sqlite3.connect('facereg.db')
    cursor = conn.cursor()
    
    existing_dates = fetch_dates()
    if date not in existing_dates:
        add_date_column(date)
    
    cursor.execute(f'''
    INSERT OR REPLACE INTO attendance (id, name, "{date}")
    VALUES (?, ?, "Present")
    ''', (id, name))
    
    conn.commit()
    conn.close()
    messagebox.showinfo("Success", f"Attendance Marked for {name} on {date}")
   
def save(id, encoded_face, name, age, subject, registration_time):
    conn = sqlite3.connect('facereg.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS registration (
        id TEXT Primary Key,
        encode BLOB NOT NULL,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        subject TEXT NOT NULL,
        registration_time TEXT NOT NULL
    )
    ''')
    cursor.execute('''
    INSERT INTO registration (id, encode, name, age, subject, registration_time)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (id, encoded_face, name, age, subject, registration_time))
    conn.commit()
    conn.close()
    
    add_attendance(id, name)
    
    messagebox.showinfo("Success", "Congratulations " + name + ", you registered successfully.")

def get_face_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = fac_mesh.process(rgb_image)
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
    return landmarks

def face_encoding(landmarks, check):
    if landmarks is None:
        if check == 0:
            messagebox.showinfo("Warning", "Invalid Face encoding")
        else:
            warning_encoding.place(relx=0.5, rely=1.0, anchor='s', y=-10)
        return None
    else:
        if check == 1:
            warning_encoding.place_forget()

        landmarks_array = np.array(landmarks, dtype=np.float32)
        flattened_landmarks = landmarks_array.flatten()

        # Calculate mean and standard deviation
        mean = np.mean(flattened_landmarks)
        std_dev = np.std(flattened_landmarks)

        # Apply Z-score normalization
        if std_dev == 0 or std_dev < 1e-6:
            normalized_landmarks = np.zeros_like(flattened_landmarks)
        else:
            normalized_landmarks = (flattened_landmarks - mean) / std_dev

        min_val = np.min(normalized_landmarks)
        max_val = np.max(normalized_landmarks)
        scaled_landmarks = (normalized_landmarks - min_val) / (max_val - min_val)

        fixed_size = 1404
        #if len(scaled_landmarks) > fixed_size:
            #scaled_landmarks = scaled_landmarks[:fixed_size]
        #elif len(scaled_landmarks) < fixed_size:
            #scaled_landmarks = np.pad(scaled_landmarks, (0, fixed_size - len(scaled_landmarks)))
        encode=np.frombuffer(scaled_landmarks,dtype="float32")
        encode=np.linalg.norm(encode)
        print(encode.shape)
        return scaled_landmarks.tobytes()

def register(img, num_faces):
    main_window.withdraw()
    if num_faces == 0:
        messagebox.showinfo("Info", "No face detected.")
        main_window.deiconify()
    elif num_faces > 1:
        messagebox.showwarning("Warning", "Multiple faces detected. Only one face can be registered at a time.")
        main_window.deiconify()
    else:
        face_landmarks = get_face_landmarks(img)
        encoded_face = face_encoding(face_landmarks,0)
        if encoded_face is not None:
            def on_submit():
                id = id_entry.get()
                name = name_entry.get()
                age = int(age_entry.get())
                subject = subject_entry.get()

                registration_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
                check_duplicates(id, encoded_face, name, age, subject, registration_time)
                input_window.destroy()
                main_window.destroy()

            global input_window
            input_window = Toplevel(root)
            input_window.title("Registration")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((500, 400), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            img_label = tk.Label(input_window, image=imgtk)
            img_label.image = imgtk 
            img_label.pack()

            back_button = tk.Button(input_window, text="←", font=("Helvetica", 14, "bold"), width=5, height=1, command=to_home, fg="White", bg="#FF6F6F")
            back_button.place(x=0, y=0)

            id_label = tk.Label(input_window, text="ID")
            id_label.pack()
            id_entry = tk.Entry(input_window)
            id_entry.pack()
            
            name_label = tk.Label(input_window, text="Name")
            name_label.pack()
            name_entry = tk.Entry(input_window)
            name_entry.pack()

            age_label = tk.Label(input_window, text="Age")
            age_label.pack()
            age_entry = tk.Entry(input_window)
            age_entry.pack()

            subject_label = tk.Label(input_window, text="Subject of Intern")
            subject_label.pack()
            subject_entry = tk.Entry(input_window)
            subject_entry.pack()

            submit_button = tk.Button(input_window, text="Submit", command=on_submit)
            submit_button.pack()


def update_frame(window, panel, label, check):
    global video, running, image, num_faces, attendance
    if check == 0:
        if not running:
            return

        ret, frame = video.read()
        if not ret:
            messagebox.showinfo("Error", "Error capturing frame")
            video.release()
            return

        frame = cv2.resize(frame, (700, 540))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)
        num_faces = 0
        if results.detections:
            num_faces = len(results.detections)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box  # Fixed attribute
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)

        label.config(text=f"Number of faces: {num_faces}")

        image = frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        panel.imgtk = imgtk
        panel.config(image=imgtk)

        panel.after(30, lambda: update_frame(window, panel, label, check))
    elif check == 1:
        if warning_face:
            warning_face.place_forget()
        if match_label:
            match_label.place_forget()
        if warning_encoding:
            warning_encoding.place_forget()
        if not running:
            return

        ret, frame = video.read()
        if not ret:
            messagebox.showinfo("Error", "Error capturing frame")
            video.release()
            return

        frame = cv2.resize(frame, (700, 540))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_frame)
        num_faces = 0
        if results.detections:
            num_faces = len(results.detections)
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box 
                ih, iw, _ = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        label.config(text=f"Number of faces: {num_faces}")

        image = frame
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        panel.imgtk = imgtk
        panel.config(image=imgtk)

        if num_faces == 0:
            if warning_face:
                warning_face.place(relx=0.5, rely=1.0, anchor='s', y=-10)
        elif num_faces > 1:
            if attendance:
                attendance.place(relx=0.5, rely=1.0, anchor='s', y=-10)
        else:
            attendance.place_forget()
            warning_face.place_forget()
            mark_attendance(image)

        panel.after(30, lambda: update_frame(window, panel, label, check))
    else:
        messagebox.showinfo("Invalid Input")
        quit_application()

def mark_attendance(img):
    if root:
        root.withdraw()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_landmarks = get_face_landmarks(rgb_img)
    encoded_face = face_encoding(face_landmarks, 1)
    if encoded_face is None:
        warning_face.place(relx=0.5, rely=1.0, anchor='s', y=-10)
        return
    
    data = fetch_students()
    if not data:
        messagebox.showinfo("Info", "No registered faces in the database.")
        return

    match_found = False
    matched_name = None
    matched_id = None
    threshold = 0.6

    encoded_face_array = np.frombuffer(encoded_face, dtype=np.float32)

    for row in data:
        id, name = row
        stored_encoding = np.frombuffer(fetch_encoded_face(id), dtype=np.float32)
        distance = np.linalg.norm(encoded_face_array - stored_encoding)
        if distance < threshold:
            match_found = True
            matched_name = name
            matched_id = id
            break
        print(distance)
        
    if match_found:
        attendance_date = date.today().strftime("%Y-%m-%d")
        save_attendance(matched_id, matched_name, attendance_date)
    else:
        match_label.place(relx=0.5, rely=1.0, anchor='s', y=-10)

def fetch_encoded_face(id):
    conn = sqlite3.connect("facereg.db")
    cursor = conn.cursor()
    cursor.execute("SELECT encode FROM registration WHERE id = ?", (id,))
    encode = cursor.fetchone()
    conn.close()
    return encode[0] if encode else None

def quit_application(event=None):
    if messagebox.askokcancel("Exit", "Closing Application"):
        if video:
            global running
            running = False
            video.release()
        root.destroy()

def open_main_window():
    global video, panel, numofface, main_window
    main_window = Toplevel(root)
    main_window.title("Face Detection")
    main_window.geometry("700x540")
    
    panel = tk.Label(main_window)
    panel.place(x=0, y=0, relwidth=1, relheight=1)

    numofface = tk.Label(main_window, text="Number of faces: 0", font=("Helvetica", 16), width=20, height=1, bg='#F5F5F5', fg="Green")
    numofface.place(relx=0.5, y=0, anchor='n')  

    back_button = tk.Button(main_window, text="←", font=("Helvetica", 14, "bold"), width=5, height=1, command=to_home, fg="White", bg="#FF6F6F")
    back_button.place(x=0, y=0)

    register_button = tk.Button(main_window, text="Register", font=("Arial", 14, "bold"), command=lambda: register(image, num_faces), fg="White", bg="Green")
    register_button.place(relx=0.5, rely=1.0, anchor='s', y=-10)

    video = cv2.VideoCapture(0)  

    if not video.isOpened():
        print("Error: Could not open video stream.")
        messagebox.showerror("Error", "Could not open video stream.")

    update_frame(main_window, panel, numofface, 0)


    

def open_attendance_window():
    global video, panel1, numofface1, attendance_window, attendance, warning_face, warning_encoding, match_label
    attendance_window = Toplevel(root)
    attendance_window.title("Attendance")
    attendance_window.geometry("700x540")
    
    panel1 = tk.Label(attendance_window)
    panel1.place(x=0, y=0, relwidth=1, relheight=1)
    
    
    back_button1 = tk.Button(attendance_window, text="←", font=("Helvetica", 14, "bold"), width=5, height=1, command=exit_attendance, fg="White", bg="#FF6F6F")
    back_button1.place(x=0, y=0)

    numofface1 = tk.Label(attendance_window, text="Number of faces: 0", font=("Helvetica", 16), width=20, height=1, bg='#F5F5F5', fg="Green")
    numofface1.place(relx=0.5, y=0, anchor='n')

    attendance = tk.Label(attendance_window, text="Only 1 attendance will be marked at one time", font=("Helvetica", 16), width=50, height=1, bg='#F5F5F5', fg="Green")
    warning_face = tk.Label(attendance_window, text="No Face Detected", font=("Helvetica", 16), width=30, height=1, bg='#F5F5F5', fg="Green")
    warning_encoding = tk.Label(attendance_window, text="Invalid Face encoding", font=("Helvetica", 16), width=30, height=1, bg='#F5F5F5', fg="Green")
    match_label=tk.Label(attendance_window, text="No Match Found", font=("Helvetica", 16), width=30, height=1, bg='#F5F5F5', fg="Green")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open video stream.")
        messagebox.showerror("Error", "Could not open video stream.")

    update_frame(attendance_window, panel1, numofface1, 1)

def exit_attendance(event=None):
    global attendance_window
    if messagebox.askokcancel("Back","Exiting Attendance phase"):
        if attendance_window:
            attendance_window.destroy()
    root.deiconify()

def to_home(event=None):
    global main_window
    if messagebox.askokcancel("Back", "Exiting registration phase"):
        if main_window:
            main_window.destroy()
 

def authentication(username_entry, pass_entry):
    username = username_entry.get()
    password = pass_entry.get()

    if username == "" or password == "":
        messagebox.showinfo("Error", "Username or Password Cannot be empty")
        return False
    elif username == "Talal" and password == "0361":
        messagebox.showinfo("Success", "Authentication Successful")
        username_entry.delete(0, tk.END)
        pass_entry.delete(0, tk.END)
        return True
    else:
        messagebox.showinfo("Error", "Invalid Username or Password")
        username_entry.delete(0, tk.END)
        pass_entry.delete(0, tk.END)
        return False

def open_home_window():
    def on_login():
        if authentication(username_entry, pass_entry):
            dashboard()

    def dashboard():
        username_label.pack_forget()
        username_entry.pack_forget()
        pass_label.pack_forget()
        pass_entry.pack_forget()
        login_button.pack_forget()
        close_button.pack_forget()
        reg_button.pack(pady=7)
        attend_button.pack(pady=7)
        download_attendance.pack(pady=7)
        logout_button.pack(pady=7)
    def logout():
        if messagebox.askokcancel("Logout","Logging Out"):
            reg_button.pack_forget()
            attend_button.pack_forget()
            logout_button.pack_forget()
            download_attendance.pack_forget()
            label1.pack(pady=20)
            username_label.pack()
            username_entry.pack()
            pass_label.pack()
            pass_entry.pack()
            login_button.pack(pady=20)
            close_button.pack(pady=5)
    global root

    root = tk.Tk()
    root.title("Home")
    root.geometry("400x300")

    label1 = tk.Label(root, text="Attendance System", font=("Helvetica", 22))
    label1.pack(pady=20)

    username_label = tk.Label(root, text="Enter Username", font=("Arial", 13))
    username_label.pack()
    username_entry = tk.Entry(root, font=("Arial", 10))
    username_entry.pack()
    
    pass_label = tk.Label(root, text="Enter Password", font=("Arial", 13))
    pass_label.pack()
    pass_entry = tk.Entry(root, font=("Arial", 10), show='*')
    pass_entry.pack()

    login_button = tk.Button(root, text="Login as Admin", command=on_login, width=20, height=2, bg="#4A87A5", fg="white")
    login_button.pack(pady=20)

    close_button = tk.Button(root, text="Close", font=("Arial", 12, "bold"), command=quit_application, width=8, height=1, bg="Red", fg="White")
    close_button.pack(pady=5)

    reg_button = tk.Button(root, text="Registration", command=open_main_window, width=20, height=2)
    reg_button.pack_forget()

    attend_button = tk.Button(root, text="Attendance", command=open_attendance_window, width=20, height=2)
    attend_button.pack_forget()

    download_attendance=tk.Button(root, text="Download Attendance", font=("Arial", 10, "bold"), command=download, width=15, height=2,bg="green",fg="white")
    download_attendance.pack_forget()
    logout_button = tk.Button(root, text="Logout",font=("Arial", 12, "bold"), command=logout, width=8, height=1,bg="red",fg="white")
    logout_button.pack_forget()

    root.mainloop()

open_home_window()
