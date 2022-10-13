import os
import requests
import sys

if __name__ == "__main__":
    dir_path = '/models/'
    r = requests.get('https://sandbox.zenodo.org/api/deposit/depositions', params={'access_token': sys.argv[1]})
    for file in r.json()[0]['files']:
        print(file['filename'])
        response = requests.get(file['links']['download'])

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = os.path.join(dir_path, file['filename'])
        with open(path, 'wb') as fp:
            fp.write(response.content)
