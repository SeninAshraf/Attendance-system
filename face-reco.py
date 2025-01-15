from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from datetime import date
import cv2, os, numpy

ui, _ = loadUiType('face-reco.ui')

class MainApp(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.tabWidget.setCurrentIndex(0)
        self.LOGINBUTTON.clicked.connect(self.login)
        self.LOGOUTBUTTON.clicked.connect(self.logout)
        self.CLOSEBUTTON.clicked.connect(self.close_window)
        self.TRAINLINK1.clicked.connect(self.show_training_form)
        self.ATTLINK1.clicked.connect(self.show_attendance_form)
        self.REPORTSLINK1.clicked.connect(self.show_report_form)
        self.TRAININGBACK.clicked.connect(self.show_mainform)
        self.ATTENDANCEBACK.clicked.connect(self.show_mainform)
        self.REPORTSBACK.clicked.connect(self.show_mainform)
        self.TRAININGBUTTON.clicked.connect(self.start_training)
        self.RECORD_2.clicked.connect(self.record_attendance)
        self.dateEdit.setDate(date.today())
        self.dateEdit.dateChanged.connect(self.selected_date)
        self.tabWidget.setStyleSheet("QTabWidget::pane{border:0;}")

        try:
            con = sqlite3.connect("face_reco.db")
            con.execute("CREATE TABLE IF NOT EXISTS attendance(attendanceid INTEGER,name TEXT,attendancedate TEXT)")
            con.commit()
            print("Table created successfully")
        except Exception as e:
            print(f"Error in database: {e}")

    ### LOGIN PROCESS ###
    def login(self):
        pw = self.PASSWORD.text()
        if pw == "123":
            self.PASSWORD.setText("")
            self.tabWidget.setCurrentIndex(1)
        else:
            self.LOGININFO.setText("Invalid Password...")
            self.PASSWORD.setText("")

    ### LOG OUT PROCESS ###
    def logout(self):
        self.tabWidget.setCurrentIndex(0)

    ### CLOSE WINDOW PROCESS ###
    def close_window(self):
        self.close()

    ### SHOW MAIN FORM ###
    def show_mainform(self):
        self.tabWidget.setCurrentIndex(1)

    ### SHOW TRAINING FORM ###
    def show_training_form(self):
        self.tabWidget.setCurrentIndex(2)

    ### SHOW ATTENDANCE FORM ###
    def show_attendance_form(self):
        self.tabWidget.setCurrentIndex(3)

    ### SHOW REPORT FORM ###
    def show_report_form(self):
        self.tabWidget.setCurrentIndex(4)
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance")
        result = cursor.fetchall()
        r = 0
        c = 0
        for row_number, row_data in enumerate(result):
            r += 1
            c = 0
            for column_number, data in enumerate(row_data):
                c += 1
        self.REPORTS.setColumnCount(c)
        for row_number, row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number, column_number, QTableWidgetItem(str(data)))
        self.REPORTS.setHorizontalHeaderLabels(['Id', 'Name', 'Date'])
        self.REPORTS.setColumnWidth(0,50)
        self.REPORTS.setColumnWidth(1,60)
        self.REPORTS.setColumnWidth(2,100)
        self.REPORTS.verticalHeader().setVisible(False)

    ### SHOW SELECTED DATE REPORT ###
    def selected_date(self):
        self.REPORTS.setRowCount(0)
        self.REPORTS.clear()
        con = sqlite3.connect("face-reco.db")
        cursor = con.execute("SELECT * FROM attendance WHERE attendancedate = '"+ str((self.dateEdit.date()).toPyDate())+"'")
        result = cursor.fetchall()
        r = 0
        c = 0
        for row_number, row_data in enumerate(result):
            r += 1
            c = 0
            for column_number, data in enumerate(row_data):
                c += 1
        self.REPORTS.setColumnCount(c)
        for row_number, row_data in enumerate(result):
            self.REPORTS.insertRow(row_number)
            for column_number, data in enumerate(row_data):
                self.REPORTS.setItem(row_number, column_number, QTableWidgetItem(str(data)))
        self.REPORTS.setHorizontalHeaderLabels(['Id', 'Name', 'Date'])
        self.REPORTS.setColumnWidth(0,50)
        self.REPORTS.setColumnWidth(1,60)
        self.REPORTS.setColumnWidth(2,100)
        self.REPORTS.verticalHeader().setVisible(False)

    ### TRAINING PROCESS ###
    def start_training(self):
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'
        sub_data = self.TraineName.text()
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
            print("The new directory is created")
            (width, height) = (130, 100)
            face_cascade = cv2.CascadeClassifier(haar_file)
            webcam = cv2.VideoCapture(0)
            count = 1
            while count < int(self.TraineCount.text()) + 1:
                print(count)
                (_, im) = webcam.read()
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite('%s/%s.png' % (path, count), face_resize)
                count += 1
                cv2.imshow('OpenCV', im)
                key = cv2.waitKey(10)
                if key == 27:
                    break
            webcam.release()
            cv2.destroyAllWindows()
            path = ""
            QMessageBox.information(self, "Attendance System", "Training Completed Successfully")
            self.TraineName.setText("")
            self.TraineCount.setText("100")

    ### RECORD ATTENDANCE ###
    def record_attendance(self):
        self.currentprocess.setText("Process started.. Waiting..")
        haar_file = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haar_file)
        datasets = 'datasets'
        (images, labels, names, id) = ([], [], {}, 0)

        # Read images and labels from dataset
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + "/" + filename
                    label = id
                    images.append(cv2.imread(path, 0))
                    labels.append(int(label))
                id += 1
        (images, labels) = [numpy.array(lis) for lis in [images, labels]]
        print(images, labels)
        (width, height) = (130, 100)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, labels)

        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():  # Check if webcam opens successfully
            print("Error: Webcam not found.")
            self.currentprocess.setText("Error: Webcam not found.")
            return  # Exit if webcam is not opened

        cnt = 0
        detected = False  # Flag to track if face has been detected and attendance registered
        unknown_detected = False  # Flag to check if an unknown face is detected

        while True:
            (_, im) = webcam.read()
            if im is None:  # If the frame is None, break the loop
                print("Error: Failed to capture image from webcam.")
                self.currentprocess.setText("Error: Failed to capture image.")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                prediction = model.predict(face_resize)
                cv2.rectangle(im, (x, y), (x + h, y + h), (0, 255, 0), 3)

                # Check the confidence score (prediction[1])
                if prediction[1] < 800:  # This threshold can be adjusted
                    # Recognize the person from the dataset
                    cv2.putText(im, '%s-%.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
                    print(names[prediction[0]])
                    self.currentprocess.setText("Detected face " + names[prediction[0]])

                    # Set attendance ID
                    attendanceid = 0
                    available = False

                    try:
                        connection = sqlite3.connect("face-reco.db")
                        cursor = connection.execute("SELECT MAX(attendanceid) from attendance")
                        result = cursor.fetchall()
                        if result and result[0][0] is not None:
                            attendanceid = int(result[0][0]) + 1
                        else:
                            attendanceid = 1  # Initialize to 1 if the table is empty
                    except sqlite3.Error as e:
                        attendanceid = 1  # Fallback to 1 if there was an error
                        print(f"Error: {e}")
                    print(attendanceid)

                    try:
                        con = sqlite3.connect("face-reco.db")
                        cursor = con.execute("SELECT * FROM attendance WHERE name='" + str(names[prediction[0]]) + "' and attendancedate = '" + str(date.today()) + "'")
                        result = cursor.fetchall()
                        if result:
                            available = True
                        if not available:
                            con.execute("INSERT INTO attendance VALUES(" + str(attendanceid) + ",'" + str(names[prediction[0]]) + "','" + str(date.today()) + "')")
                            con.commit()
                    except sqlite3.Error as e:
                        print(f"Error in database insert: {e}")
                    finally:
                        if con:
                            con.close()

                    print("Attendance Registered Successfully")
                    self.currentprocess.setText("Attendance entered for " + names[prediction[0]])
                    detected = True  # Set detected flag to True
                    cnt = 0
                    break  # Exit the loop after registering attendance for the first detected face
                else:
                    # If prediction confidence is too low, label as "Unknown"
                    cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                    if not unknown_detected:
                        print("Unknown person detected!")
                        self.currentprocess.setText("Unknown Person detected!")
                        cv2.imwrite('unknown.jpg', im)
                        unknown_detected = True

            if detected:
                break  # Exit the outer while loop once attendance has been registered
            elif unknown_detected:
                break  # Exit if an unknown person was detected
            else:
                cv2.putText(im, 'Searching...', (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

            cv2.imshow("Face Recognition", im)
            key = cv2.waitKey(10)
            if key == 27:  # If ESC key is pressed, break the loop
                break

        webcam.release()
        cv2.destroyAllWindows()

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
