from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineCore import *
from PyQt5.QtWebEngineWidgets import *
import sys

class QtSchemeHandler(QWebEngineUrlSchemeHandler):
    def requestStarted(self, job):
        request_url = job.requestUrl()
        request_path = request_url.path()
        file = QFile('.' + request_path)
        file.setParent(job)
        job.destroyed.connect(file.deleteLater)
        file_info = QFileInfo(file)
        mime_database = QMimeDatabase()
        mime_type = mime_database.mimeTypeForFile(file_info)
        job.reply(mime_type.name().encode(), file)


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setStyleSheet('background-color: blue;')
        self.verticalLayout = QVBoxLayout()
        self.setLayout(self.verticalLayout)
        self.browser = QWebEngineView()
        self.scheme_handler = QtSchemeHandler()
        self.browser.page().profile().installUrlSchemeHandler(
            b"qt", self.scheme_handler
        )
        url = QUrl("qt://main")
        url.setPath("/index.html")
        self.browser.load(url)
        self.verticalLayout.addWidget(self.browser)
        self.browser.loadFinished.connect(self.show)


if __name__ == "__main__":
    scheme = QWebEngineUrlScheme(b"qt")
    scheme.setFlags(QWebEngineUrlScheme.CorsEnabled)
    QWebEngineUrlScheme.registerScheme(scheme)
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())