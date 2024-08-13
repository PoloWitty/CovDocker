"""
desc:	This script is used to get the pdb2mechanism.csv file from the website
        https://drug-discovery.vm.uni-freiburg.de/covpdb/mechanisms_list/initial=Allsortedby=mechanism_name
        which contains the information of the mechanism nameand the corresponding pdb id 
author:	Yangzhe Peng
date:	2024/08/03
"""


import requests
from bs4 import BeautifulSoup
import tqdm
import time

# get the related info from https://drug-discovery.vm.uni-freiburg.de/covpdb/mechanisms_list/initial=Allsortedby=mechanism_name
# // 初始化一个数组来存储链接
# const links = [];

# // 遍历1到21的数字
# for (let i = 1; i <= 21; i++) {
#     // 构造选择器字符串
#     const selector = `body > div:nth-child(14) > table > tbody > tr:nth-child(${i}) > td:nth-child(7) > a`;
#     // 使用选择器查找a元素
#     const a = document.querySelector(selector);
#     // 如果a元素存在，获取其链接并添加到数组中
#     if (a) {
#         links.push(a.text);
#     }
# }

# // 输出链接数组
# console.log(links);

mechanisms = ['Aziridine Opening', 'Beta-Lactam Addition', 'Borylation', 'Composite Reaction', 'Cyclohemiaminoacetalization', 'Disulfide Formation', 'Epoxide Opening', 'Hemiaminalization', 'Hemi(thio)acetalization', 'Imidazolidinone Opening', 'Imine Condensation', 'Lactone Addition', 'Michael Addition', 'Nucleophilic Acyl Substitution', 'Nucleophilic Addition to a Double Bond', 'Nucleophilic Addition to a Triple Bond', 'Nucleophilic Aliphatic Subsititution', 'Nucleophilic Aromatic Substitution', 'Phosphorylation', 'Ring Opening', 'Sulfonylation']
urls = [
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=21",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=11",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=6",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=8",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=4",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=16",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=3",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=20",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=1",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=28",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=19",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=29",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=5",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=13",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=7",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=10",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=18",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=17",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=9",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=14",
    "https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list_by_id/search_type=by_reaction_idsearch_id=12"
]
pdb_nums = [18, 280, 200, 42, 12, 62, 116, 13, 243, 53, 123, 29, 363, 140, 44, 100, 226, 43, 116, 41, 30]
warhead_nums = [1, 1, 3, 14, 4, 3, 1, 2, 4, 1, 4, 4, 11, 17, 13, 4, 18, 7, 6, 13, 2]
mechanism2url = dict(zip(mechanisms, urls))

def fetch_page(url):
    """请求页面内容并解析"""
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def extract_pdb_ids(soup):
    """从页面中提取pdb_id"""
    pdb_ids = []
    table = soup.find('table')  # 找到包含pdb_id的表格
    if table:
        rows = table.find_all('tr')
        for row in rows:  # 跳过表头
            cells = row.find_all('td')
            if len(cells) > 1:
                pdb_id = cells[1].text.strip()  # 根据实际表格结构调整索引
                pdb_ids.append(pdb_id)
    return pdb_ids

if __name__ == "__main__":
    fetch_warhead = True # false: pdb2mechanism; true: warhead2mechanism
    
    if fetch_warhead:
        output_file = "./data/auxiliary/covpdb_warhead2mechanism.csv"
    else:
        output_file = "./data/auxiliary/covpdb_pdb2mechanism.csv"

    with open(output_file, "w") as fp:
        # write header
        if fetch_warhead:
            fp.write("mechanism,warhead\n")
        else:
            fp.write("mechanism,pdb_id\n")

        for i, mechanism in enumerate(mechanisms):
            print(f'Fetching mechanism {mechanism}...')
            page_number = 1
            all_pdb_ids = []

            while True:
                # 构造分页URL
                base_url = urls[i].replace('complexes_list_by_id','warheads_list_by_id') if fetch_warhead else urls[i]
                url = f'{base_url}?page={page_number}'
                print(f'Fetching page {page_number}...')
                soup = fetch_page(url)
                pdb_ids = extract_pdb_ids(soup)

                all_pdb_ids.extend(pdb_ids)
                if fetch_warhead:
                    if len(all_pdb_ids) == warhead_nums[i]:
                        break
                else:
                    if len(all_pdb_ids) == pdb_nums[i]:
                        break
                
                page_number += 1
                time.sleep(1)  # 暂停1秒以避免过于频繁的请求

            for pdb_id in all_pdb_ids:
                fp.write(f"{mechanisms[i]},{pdb_id}\n")