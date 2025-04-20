# Upload ZIP Report Bundle to Google Drive via PyDrive (Public Sharing)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

BUNDLE_PATH = "symbolic_model_report_bundle.zip"

# Authenticate and create PyDrive client
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

# Upload file and make public
def upload_file_to_drive(filepath, drive):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    file_drive = drive.CreateFile({'title': os.path.basename(filepath)})
    file_drive.SetContentFile(filepath)
    file_drive.Upload()

    # Make file public
    file_drive.InsertPermission({
        'type': 'anyone',
        'value': 'anyone',
        'role': 'reader'
    })

    print(f"☁️ Uploaded to Google Drive: {file_drive['title']}")
    print(f"🔗 Public link: https://drive.google.com/file/d/{file_drive['id']}/view")

if __name__ == "__main__":
    drive = authenticate_drive()
    upload_file_to_drive(BUNDLE_PATH, drive)
