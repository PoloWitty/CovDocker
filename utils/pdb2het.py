"""
desc:	This script is used to get the pdb2het.csv file from the website
        https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list/initial=Allsortedby=protein_id?page=1
        which contains the information of the pdb id and the corresponding het code (can be used the get the covalent ligand information from the complex pdb file)
author:	Yangzhe Peng
date:	2023/12/22
"""


import requests
from bs4 import BeautifulSoup
import tqdm


if __name__ == "__main__":
    output_file = "pdb2het.csv"

    with open(output_file, "w") as fp:
        # write header
        fp.write("pdb_id,het_code\n")

        # there are 92 pages in total
        for page in tqdm.trange(1,93,desc='parsing the webpages'):
            url = f"https://drug-discovery.vm.uni-freiburg.de/covpdb/complexes_list/initial=Allsortedby=protein_id?page={page}"

            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            table = soup.find("table")
            rows = table.find_all("tr")

            for row in rows:
                cells = row.find_all("td")
                
                pdb_id = cells[1].text.strip()
                het_code = cells[-3].text.strip()
                fp.write(f"{pdb_id},{het_code}\n")