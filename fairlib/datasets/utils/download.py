import os
import requests
from tqdm.auto import tqdm

def download(url: str, dest_folder: str, chunk_size: int = 10*1024*1024):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    try:
        total_length = int(r.headers.get('content-length'))
    except:
        total_length = 0

    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=(total_length//chunk_size)+1):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))