import os
import requests
from concurrent.futures import ThreadPoolExecutor


def download_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

if __name__ == '__main__':
    urls = []
    for i in range(100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=SaiedAlshahrani%2FEgyptian_Arabic_Wikipedia_20230101&config=SaiedAlshahrani--Egyptian_Arabic_Wikipedia_20230101&split=train&offset={i}&limit=100"
        urls.append(url)

    if not os.path.exists('arabic_egyptian'):
        os.makedirs('arabic_egyptian')

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, url in enumerate(urls):
            future = executor.submit(download_url, url)
            futures.append((i, future))

        for i, future in futures:
            result = future.result()
            if result is not None:
                local_filename = f"arabic_egyptian/data{i}.txt"
                with open(local_filename, 'w', encoding='utf-8') as file:
                    file.write(result)
                print("Download successful. Content saved as", local_filename)
            else:
                print("Error occurred while downloading content for URL", urls[i])