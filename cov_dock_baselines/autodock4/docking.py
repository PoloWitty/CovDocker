import os
import multiprocessing
import tqdm
import pandas as pd
import subprocess
import argparse
import json

def run(cmd):
    subprocess.run(cmd, shell=True, check=True, timeout=60*15)

def replace_file_content(file_path, old_str, new_str):
    # Read in the file
    with open(file_path, 'r') as fp:
        filedata = fp.read()

    # Replace the target string
    filedata = filedata.replace(old_str, new_str)

    # Write the file out again
    with open(file_path, 'w') as fp:
        fp.write(filedata)

def at_close():
    # return to the original path
    os.chdir(work_dir_abs)

def single_docking(x):
    i,row = x
    pdb_id = row['pdb_id']
    name, resName, chainID, resSeq = row['bond'].split('-')[0].split(' ') # SG CYS A 145-C8 T9P A 405
    bond_res = f'{chainID}:{resName}{resSeq}' 
    
    # os.chdir(os.path.abspath(os.path.dirname(__file__))+f"/data/processed/bonded/{pdb_id}/")
    os.chdir(work_dir_abs+f"/bonded/{pdb_id}/")
    # print(os.getcwd())
    
    # if pdb_id not in ['7B4E', '7JN7', '6YEO', '6XRO', '7NXW', '7BDT', '6XY5', '7AU0', '7NIZ', '7LYH', '7BA7', '6XB2', '7E9A', '7KRF', '6WYG', '6YNQ', '6WZV', '7BJW', '7BFW', '7RC0', '7AYF', '6Y1L', '7BAB', '7ONK', '7A1W', '6YPD', '6Z25', '7ATX', '6Y1N', '7A72', '6Y49', '7AU1', '7AU8', '6VUA', '6X1M', '7NJ9', '7BJB', '7JPZ', '6YCC', '6X6C', '7AZ1', '6XCC', '7B49', '7B4N', '7BAA', '7BIQ', '7LNR', '7DNC', '7LY1', '7ATW', '6XXC', '6ZBX', '6VHS', '6VIM', '5RFG']:
    #     return True
    # if pdb_id != '5RG3':
    #     return True
    
    # if pdb_id in ['7AWE', '7KRF', '7FD5', '6YPD', '6Y1N', '6Y49', '7FD4', '7KRC', '6YEN']:
    #     return pdb_id # failed when preparing docking data
    
    try:
        with open(f'./{pdb_id}_ligindices.txt', 'r') as fp:
            ligindices = fp.read()
    except:
        at_close()
        return pdb_id
    
    # step 0) remove alt conf
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_pdb_split_alt_confs.py -r ./{pdb_id}_10Apocket.pdb -o {pdb_id}_10Apocket
    '''
    try:
        run(generate_cmd)
        receptor = f'{pdb_id}_10Apocket_A'
        if not os.path.exists(receptor+'_A.pdb'): # Nothing to do:no alt loc atoms
            receptor = f'{pdb_id}_10Apocket'
    except:
        at_close()
        return pdb_id      

    # round (1). step 1) align
    # output ligcovalent.pdb
    align_cmd = f'''
    $MGLROOT/bin/pythonsh $AdCovScriptRoot/prepareCovalent.py --ligand ./{pdb_id}_ligand.autodock_randomConform.sdf \
                --ligindices {ligindices}\
                --receptor {receptor}.pdb \
                --residue {bond_res}\
                --outputfile ligcovalent.pdb \
    '''
    try:
        run(align_cmd)
    except:
        at_close()
        return pdb_id

    # round (2). step 1) align for gt ligand
    # output ligcovalent.pdb
    align_cmd = f'''
    $MGLROOT/bin/pythonsh $AdCovScriptRoot/prepareCovalent.py --ligand ./{pdb_id}_ligand.autodock_gt.sdf \
                --ligindices {ligindices}\
                --receptor {receptor}.pdb \
                --residue {bond_res}\
                --outputfile ligcovalent_gt.pdb \
    '''
    try:
        run(align_cmd)
    except:
        at_close()
        return pdb_id


    # round (1). step 2) generate pdbqt file for autodock
    # for receptor
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r {receptor}.pdb -A hydrogens
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    # for ligand
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ./ligcovalent.pdb
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    
    # round (2). step 2) generate pdbqt file for autodock gt ligand
    # for ligand
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r ./ligcovalent_gt.pdb 
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    

    # round (1). step 3) generate flexible/rigid pdbqt file
    # for receptor
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py -r {receptor}.pdbqt -s {receptor}:{bond_res}
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    # for ligand
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py -r ./ligcovalent.pdbqt -s ligcovalent:{bond_res}
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    
    
    # round (2). step 3) generate flexible/rigid pdbqt file for gt ligand
    # for ligand
    generate_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_flexreceptor4.py -r ./ligcovalent_gt.pdbqt -s ligcovalent_gt:{bond_res}
    '''
    try:
        run(generate_cmd)
    except:
        at_close()
        return pdb_id
    

    # generate the flex ligand pdb file as target of dock result
    convert_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/pdbqt_to_pdb.py -f ./ligcovalent_gt_flex.pdbqt -o ligcovalent_gt_flex.pdb
    '''
    try:
        run(convert_cmd)
    except:
        at_close()
        return pdb_id
    
    # step 4) generate config file for autogrid and autodock
    # for autogrid
    config_cmd=f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_gpf4.py -r ./{receptor}_rigid.pdbqt\
            -x ligcovalent_flex.pdbqt\
            -l ligcovalent_flex.pdbqt\
            -y -I 20\
            -o autogrid_config.gpf
    '''
    try:
        run(config_cmd)
    except:
        at_close()
        return pdb_id
    # for autodock
    config_cmd=f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_dpf4.py -r {receptor}_rigid.pdbqt\
            -x ligcovalent_flex.pdbqt\
            -l ligcovalent_flex.pdbqt\
            -o autodock_config.dpf\
            -p move='empty'
    '''
    try:
        run(config_cmd)
    except:
        at_close()
        return pdb_id
    run('touch empty') # dock the covalent ligand as a flexible residue and ignore any 'true' ligand ("move='empty'")
    # manually edited to set the appropriate energy model so that the docking score corresponds to the interaction between the flexible residue (the ligand) and the rigid receptor
    replace_file_content('autodock_config.dpf', 'unbound_model bound', 'unbound_energy 0.0')
    replace_file_content('autodock_config.dpf', 'unbound_model extended', 'unbound_energy 0.0')
    
    # step 5) run autogrid and autodock
    # autogrid
    run_cmd = f'''
    $AutoDockRoot/autogrid4 -p autogrid_config.gpf -l {receptor}.glg
    '''
    try:
        run(run_cmd)
    except:
        at_close()
        return pdb_id
    # autodock
    run_cmd = f'''
    $AutoDockRoot/autodock4 -p autodock_config.dpf -l result.dlg
    '''
    try:
        run(run_cmd)
    except:
        at_close()
        return pdb_id
    
    # step 6) convert output dlg file to sdf
    # # more convert option ref to https://openbabel.org/docs/FileFormats/AutoDock_PDBQT_format.html (add them after -a(read) -x(write))
    # convert_cmd = f'''
    # obabel -ipdbqt result.dlg -opdb -O result.pdb -ad
    # '''
    convert_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/write_lowest_energy_ligand.py -f result.dlg -o result.pdbqt
    '''
    try:
        run(convert_cmd)
    except:
        at_close()
        return pdb_id
    convert_cmd = f'''
    $MGLROOT/bin/pythonsh $MGLROOT/MGLToolsPckgs/AutoDockTools/Utilities24/pdbqt_to_pdb.py -f result.pdbqt -o result.pdb
    '''
    try:
        run(convert_cmd)
    except:
        at_close()
        return pdb_id
    
    # return to the original path
    os.chdir(work_dir_abs)
    return True


if __name__=='__main__':
    # make suer AdCovScriptRoot, MGLROOT, AutoDockRoot is in env
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, default='./data', help='Path to data, note that the output will be in this dir too')
    parser.add_argument("--only-unseen", type=int, default=0, help="only process unseen entry")
    args = parser.parse_args()
    
    work_dir_abs = os.path.abspath(args.work_dir)
    
    if args.only_unseen:
        df = pd.read_csv(args.work_dir+'/dataset.unseen.csv')
        df = df[df['set']=='unseen']
    else:
        df = pd.read_csv(args.work_dir+'/dataset.csv')
        df = df[df['set']=='test']
    
    # for x in df.iterrows():
    #     single_docking(x)
    #     breakpoint()
    
    error_pdb = []
    with tqdm.trange(len(df), desc='docking') as pbar:
        with multiprocessing.Pool(10) as pool:
            for inner_output in pool.imap(single_docking, df.iterrows()):
                if inner_output != True:
                    error_pdb.append(inner_output)
                pbar.update()

    print(f"failed pdb {len(error_pdb)}")
    print(error_pdb)
    
# failed pdb 38
# ['7A2A', '7AWE', '6YEO', '7NXW', '7BDT', '6XY5', '7CC2', '6ZBW', '7E9A', '7KRF', '6YPY', '6WZV', '7NWS', '7FD5', '7BA6', '7BAB', '7A1W', '6YPD', '7ATX', '6Y1N', '7A72', '6Y49', '7AU1', '6X1M', '7FD4', '7NJ9', '7BG3', '6YCC', '6X6C', '7AZ1', '7KRC', '7B4H', '7D3I', '7BIW', '7DNC', '7NXT', '6ZBX', '6YEN']