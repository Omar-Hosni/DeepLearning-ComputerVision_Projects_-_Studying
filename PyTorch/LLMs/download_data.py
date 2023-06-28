import os
import requests
from concurrent.futures import ThreadPoolExecutor


data_dir = 'D:\Projects\ML projects\Scikit-Learn\PyTorch\LLMs\data'
combined_data_dir = 'D:\Projects\ML projects\Scikit-Learn\PyTorch\LLMs\combined_data\combined.txt'



def download_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None


def read_all_data_and_combine_them(data_dir, output_dir):
    with open(output_dir, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

if __name__ == '__main__':
    urls = []
    for i in range(11):
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
                local_filename = f"{data_dir}/data{i}.txt"
                with open(local_filename, 'w', encoding='utf-8') as file:
                    file.write(result)
                print("Download successful. Content saved as", local_filename)
            else:
                print("Error occurred while downloading content for URL", urls[i])

    read_all_data_and_combine_them(data_dir, combined_data_dir)

