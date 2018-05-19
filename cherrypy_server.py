"""
Launches a simple server in port 9090 to upload a bill of lading
The document is parsed and then the result is shown
"""


import os
import os.path

import cherrypy
from cherrypy.lib import static

import tempfile
from process_bl_pdf_with_image_alignmet import parse_bl_with_alignment

localDir = os.path.dirname(__file__)
absDir = os.path.join(os.getcwd(), localDir)

config = {
    'global': {
        'server.socket_host': '0.0.0.0',
        # 'server.socket_host' : '127.0.0.1',
        'server.socket_port': 9090,
        'server.thread_pool': 8,
        'server.max_request_body_size': 0,
        'server.socket_timeout': 60
    }
}


def bootstrapTemplate(title, html_body):
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <!-- Required meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    
        <!-- Bootstrap CSS -->
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    
        <title>{title}</title>
      </head>
      <body>
        {html_body}
        <!-- Optional JavaScript -->
        <!-- jQuery first, then Popper.js, then Bootstrap JS -->
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
      </body>
    </html>
      """.format(title=title, html_body=html_body)


class FileDemo(object):

    @cherrypy.expose
    def index(self):
        return bootstrapTemplate("Pdf ocr tester", """"
            <h2>Upload a BILL OF LADING pdf file</h2>
            <form action="upload" method="post" enctype="multipart/form-data">
            filename: <input type="file" name="myFile" class="btn but-primary"/><br />
            <input type="submit" class="btn but-primary"/>
            </form>
        <!--    <h2>Download a file</h2>
            <a href='download'>This one</a>
        -->
        """)

    @cherrypy.expose
    def upload(self, myFile):
        out = """<html>
        <body>
            myFile length: %s<br />
            myFile filename: %s<br />
            myFile mime-type: %s
        </body>
        </html>"""

        if not myFile.filename.upper().endswith(".PDF"):
            raise cherrypy.HTTPError(400, 'Only pdf files are valid')

        # Although this just counts the file length, it demonstrates
        # how to read large files in chunks instead of all at once.
        # CherryPy reads the uploaded file into a temporary file;
        # myFile.file.read reads from that.
        size = 0
        with cherrypy.HTTPError.handle(Exception, 500):
            # with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = os.path.join(os.path.dirname(__file__), "pdfs")
            temp_filename = os.path.join(tmpdirname, myFile.filename)
            with open(temp_filename, "wb") as f:
                while True:
                    data = myFile.file.read(8192)
                    if not data:
                        break
                    f.write(data)
                    size += len(data)

            ocr_result = parse_bl_with_alignment(temp_filename)

            html_body = ""
            for k, v in zip(ocr_result.keys(), ocr_result.values()):
                html_body += '<div class = "row"><div class="col"><h4>{k}</h4></col><div class="col">{v}</col></div>'.format(
                    k=k, v=v)
            return bootstrapTemplate("Parsed BL", html_body)

    @cherrypy.expose
    def download(self):
        path = os.path.join(absDir, 'pdf_file.pdf')
        return static.serve_file(path, 'application/x-download',
                                 'attachment', os.path.basename(path))


if __name__ == '__main__':
    # CherryPy always starts with app.root when trying to map request URIs
    # to objects, so we need to mount a request handler root. A request
    # to '/' will be mapped to HelloWorld().index().
    cherrypy.quickstart(FileDemo(), config=config)
