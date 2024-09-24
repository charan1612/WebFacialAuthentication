from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import os
import sqlite3
import numpy as np
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)
app.secret_key = 'aVerySecretKeyThatIsVeryLongAndComplex123!'


DATABASE = 'database/users.db'
FACE_DATA_FOLDER = 'face_data/'

# Helper function to connect to the database
def get_db():
    conn = sqlite3.connect(DATABASE)
    return conn

# Helper function to save face data to the database
def save_user_to_db(username, face_data_path):
    conn = get_db()
    try:
        conn.execute('INSERT INTO users (username, face_data_path) VALUES (?, ?)', (username, face_data_path))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()
    return True

# Helper function to get user data from the database
def get_user_from_db(username):
    conn = get_db()
    cursor = conn.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user

# Root route
@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in the templates folder

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']

        # Create face_data folder if it doesn't exist
        if not os.path.exists(FACE_DATA_FOLDER):
            os.makedirs(FACE_DATA_FOLDER)

        # Capture face using webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access the webcam. Please check your webcam settings.', 'danger')
            return redirect(url_for('signup'))

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            flash('Face cascade classifier not found. Please check the file path.', 'danger')
            cap.release()
            return redirect(url_for('signup'))

        detected_face = False
        while True:
            ret, frame = cap.read()
            if not ret:
                flash('Failed to capture image. Please try again.', 'danger')
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('signup'))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                detected_face = True
                face = gray[y:y + h, x:x + w]  # Save as grayscale
                face_resized = cv2.resize(face, (200, 200))  # Ensure a consistent size for comparison
                face_path = os.path.join(FACE_DATA_FOLDER, f'{username}.jpg')
                cv2.imwrite(face_path, face_resized)  # Save the face image
                cap.release()
                cv2.destroyAllWindows()
                if save_user_to_db(username, face_path):
                    flash('Signup successful!', 'success')
                    return redirect(url_for('login'))
                else:
                    flash('Username already exists. Try a different one.', 'danger')
                    return redirect(url_for('signup'))

            cv2.imshow('Signup - Capture your face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not detected_face:
            flash('No face detected. Please try again.', 'danger')

        cap.release()
        cv2.destroyAllWindows()
    return render_template('signup.html')




# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        user = get_user_from_db(username)
        if not user:
            flash('User does not exist', 'danger')
            return redirect(url_for('login'))

        # Capture face using webcam for login
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access the webcam. Please check your webcam settings.', 'danger')
            return redirect(url_for('login'))

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        face_detected = False
        while True:
            ret, frame = cap.read()
            if not ret:
                flash('Failed to capture image. Please try again.', 'danger')
                cap.release()
                cv2.destroyAllWindows()
                return redirect(url_for('login'))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_detected = True
                face = gray[y:y + h, x:x + w]
                face_path = user[2]

                # Load stored face and compare
                stored_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
                if stored_face is None:
                    flash('Stored face image not found. Please contact support.', 'danger')
                    cap.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('login'))

                # Resize captured face to match stored face dimensions
                face_resized = cv2.resize(face, (stored_face.shape[1], stored_face.shape[0]))

                # Structural Similarity Index (SSIM) for better comparison
                similarity_index, _ = ssim(face_resized, stored_face, full=True)

                if similarity_index > 0.5:  # Adjust threshold as needed
                    flash('Login successful!', 'success')
                    cap.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('dashboard'))
                else:
                    flash('Face does not match. Try again.', 'danger')
                    cap.release()
                    cv2.destroyAllWindows()
                    return redirect(url_for('login'))

            cv2.imshow('Login - Capture your face', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if not face_detected:
            flash('No face detected. Please try again.', 'danger')

        cap.release()
        cv2.destroyAllWindows()
    return render_template('login.html')



# Route for the dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
    

if __name__ == '__main__':
    app.run(debug=True)
