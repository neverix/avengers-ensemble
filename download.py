import urllib.request
import os
import shutil
import zipfile


out_path = "datasets"


def main():
    print("Downloading...")
    file, _ = urllib.request.urlretrieve("https://russiansuperglue.com/tasks/download")
    print("Extracting...")
    zip_file = zipfile.ZipFile(file)
    zip_file.extractall('.')
    shutil.rmtree("__MACOSX")
    shutil.rmtree(out_path)
    os.rename("combined", out_path)
    print("Done!")


if __name__ == '__main__':
    main()
