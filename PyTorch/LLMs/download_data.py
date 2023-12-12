import os
import requests
from concurrent.futures import ThreadPoolExecutor


data_dir = 'D:\Projects\ML projects\Scikit-Learn\PyTorch\LLMs\data'
combined_data_dir = 'D:\Projects\ML projects\Scikit-Learn\PyTorch\LLMs\combined_data\combined.txt'



def download_and_save_url(offset, url):

    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        local_filename = os.path.join(data_dir, f'data{offset}.txt')
        with open(local_filename, 'w', encoding='utf-8') as file:
            file.write(content)
        print("Download successful. Content saved as", local_filename)
    else:
        print("Error occurred while downloading content for URL", url)

def read_all_data_and_combine_them(data_dir, output_dir):
    with open(output_dir, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())

if __name__ == '__main__':
    import os
    from concurrent.futures import ProcessPoolExecutor
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    urls = [
        f"https://datasets-server.huggingface.co/rows?dataset=SaiedAlshahrani%2F" \
        f"Egyptian_Arabic_Wikipedia_20230101&" \
        f"config=default&split=train&offset={i}&limit=100"
        for i in range(100)
    ]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use map to apply download_and_save_url to each URL and offset
        futures = executor.map(download_and_save_url, range(len(urls)), urls)

    # Combine data
    read_all_data_and_combine_them(data_dir, combined_data_dir)