import requests
import zipfile
import shutil
import os
import io


out_path = "datasets"


def main():
    print("Downloading...")
    response = requests.get("https://russiansuperglue.com/tasks/download", verify=False)
    print("Extracting...")
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    zip_file.extractall('.')
    shutil.rmtree("__MACOSX")
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.rename("combined", out_path)
    print("Done!")


if __name__ == '__main__':
    main()
