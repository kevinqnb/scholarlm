import os
import json
from scholarlm.utils import get_filenames_in_directory
from dotenv import load_dotenv
load_dotenv()

main_directory = os.getenv("POND_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")

with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)


text_files = get_filenames_in_directory(text_directory, ignore = [".DS_Store"])
text_files.sort()

n = len(text_files)
m = n // 3

data = []
for i in range(3):
    filename = f"data/pond_results_10_papers_v1_vllm_{i + 1}.json"
    offset = m * i
    with open(filename, "r") as f:
        file_data = json.load(f)
        for entry in file_data:
            entry['paper_id'] = entry['paper_id'] + offset

        data.extend(file_data)

outfile = "data/pond_results_10_papers_v1_vllm.json"
with open(outfile, "w") as f:
    json.dump(data, f, indent=4)