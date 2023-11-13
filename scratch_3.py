import sys

from PyQt5 import QtCore

loop = 0


def timerEvent():
    global time
    global app
    global loop
    print(loop)
    time = time.addSecs(1)
    print(time.toString("hh:mm:ss"))
    if loop >= 10:
        app.quit()
    loop += 1


if __name__ == "__main__":
    app = QtCore.QCoreApplication(sys.argv)
    timer = QtCore.QTimer()
    time = QtCore.QTime(0, 0, 0)
    timer.timeout.connect(timerEvent)
    timer.start(1000)
    app.exec()
