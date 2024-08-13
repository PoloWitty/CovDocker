"""
desc:	This script is used to get the pdb2mechanism.csv file for CovBinderInPDB dataset and align its mechanism names with the ones in the CovPDB dataset
author:	Yangzhe Peng
date:	2024/08/03
"""
import pandas as pd

mechanisms = ['Aziridine Opening', 'Beta-Lactam Addition', 'Borylation', 'Composite Reaction', 'Cyclohemiaminoacetalization', 'Disulfide Formation', 'Epoxide Opening', 'Hemiaminalization', 'Hemi(thio)acetalization', 'Imidazolidinone Opening', 'Imine Condensation', 'Lactone Addition', 'Michael Addition', 'Nucleophilic Acyl Substitution', 'Nucleophilic Addition to a Double Bond', 'Nucleophilic Addition to a Triple Bond', 'Nucleophilic Aliphatic Subsititution', 'Nucleophilic Aromatic Substitution', 'Phosphorylation', 'Ring Opening', 'Sulfonylation']

warhead2mechanism = {
    # page s-2
    '1,3-oxazin-6-one': 'Ring Opening',
    '3H-1,3-oxazine-2,6-dione': 'Ring Opening',
    '4-halo-α-pyrone': 'Lactone Addition',
    'Acrylamide': 'Michael Addition', # miss
    'Acrylate': 'Michael Addition', # miss
    'Acylphosphonate': 'Nucleophilic Acyl Substitution',
    # page s-3
    'Aldehyde': 'Hemi(thio)acetalization', 
    'Aldehyde_Adduct': 'Nucleophilic Aliphatic Subsititution', # miss
    'Alkene': 'Nucleophilic Addition to a Double Bond', # miss
    'Alkenyl_Halide': 'Nucleophilic Aliphatic Subsititution', # miss
    'Alkyl_Halide': 'Nucleophilic Acyl Substitution',
    'Alkyne': 'Nucleophilic Addition to a Triple Bond',
    'Allyl_Vinyl_Ether': 'Nucleophilic Acyl Substitution',
    # page s-4
    'Amide': 'Nucleophilic Acyl Substitution',
    'Amidine': 'Nucleophilic Addition to a Double Bond',
    'Aminohalophosphate': 'Phosphorylation', # miss
    'Amitrole': 'Nucleophilic Aromatic Substitution',
    'Aryl_Ether': 'Nucleophilic Aromatic Substitution', # miss and not sure
    'Aryl_Halide': 'Nucleophilic Aromatic Substitution',
    'Aryl_Sulfone': 'Nucleophilic Aromatic Substitution',
    # page s-5
    'Azanitrile': 'Nucleophilic Addition to a Triple Bond', # miss
    'Azide': 'Nucleophilic Aromatic Substitution',
    'Aziridine': 'Aziridine Opening',
    'Azo': 'Nucleophilic Addition to a Double Bond', # miss and not sure
    'Boric_Acid': 'Borylation', # miss
    'Boronic_Acid': 'Composite Reaction',
    'Butadienyl_Carbonyl': 'Michael Addition',
    # page s-6
    'Carbamate': 'Nucleophilic Acyl Substitution',
    'Carbonate': 'Nucleophilic Addition to a Double Bond',
    'Carboxylic_Acid': 'Nucleophilic Acyl Substitution', # many
    'Catechol': 'Nucleophilic Aromatic Substitution',
    'Conjugated_System': 'Nucleophilic Addition to a Double Bond', # miss
    'Cyclic_Acylphosphate': 'Phosphorylation', # not sure and miss
    # page s-7
    'Cyclic_Borate': 'Borylation', # miss
    'Cyclic_Boronic_Acid': 'Borylation', # miss
    'Cyclic_Phosphate_Ester': 'Phosphorylation', # miss
    'Cyclopropane': 'Ring Opening', # miss
    'Cyclosulfate': 'Sulfonylation', # miss
    'Disulfide': 'Disulfide Formation',
    'Enamine': 'Nucleophilic Addition to a Double Bond',
    # page s-8
    'Ene_Ynone': 'Michael Addition', # miss and not sure
    'Epoxide': 'Epoxide Opening',
    'Ester': 'Nucleophilic Addition to a Double Bond', 
    'Furan': 'Ring Opening', 
    'Glycoside': 'Nucleophilic Aliphatic Subsititution', # miss
    'Glycosyl_Halide': 'Nucleophilic Aliphatic Subsititution', # miss
    # page s-9
    'Haloacetamide': 'Nucleophilic Aliphatic Subsititution', # miss
    'Haloacetamidine': 'Nucleophilic Aliphatic Subsititution', # miss
    'Haloisoxazoline': 'Imine Condensation', # not sure and miss
    'Halophosphate': 'Phosphorylation', # miss 
    'Halophosphonate': 'Phosphorylation', # miss
    'Halosulfate': 'Sulfonylation', # miss
    # page s-10
    'Hemiacetal': 'Nucleophilic Acyl Substitution',
    'Hydrazide': 'Nucleophilic Acyl Substitution', # many
    'Hydroxylammonium': 'Imine Condensation', # miss and not sure
    'Imidazolidinone': 'Imidazolidinone Opening',
    'Isocyanide': 'Nucleophilic Addition to a Triple Bond',
    'Isothiazolinone': 'Ring Opening',
    'Isothiocyanate': 'Nucleophilic Addition to a Double Bond',
    # page s-11
    'Isoxazole': 'Nucleophilic Addition to a Double Bond', # miss
    'Isoxazolidin-3-one': 'Ring Opening', 
    'Ketone': 'Composite Reaction',
    'Mannich_Base': 'Nucleophilic Aliphatic Subsititution',
    'Nicotinamide_Riboside': 'Nucleophilic Acyl Substitution', # miss and not sure
    'Nitrile': 'Nucleophilic Aliphatic Subsititution', # many
    'Nitroalkene': 'Nucleophilic Addition to a Double Bond', # miss
    # page s-12
    'Nitroarene': 'Nucleophilic Addition to a Double Bond',
    'Nitroso': 'Nucleophilic Addition to a Double Bond', # miss and not sure
    'O-acyl_Hydroxamic_Acid': 'Nucleophilic Acyl Substitution',
    'Oxycarbonyl-disulfide ': 'Disulfide Formation', # miss
    'Phosphate_Ester': 'Phosphorylation', # miss
    'Phosphonate_Ester': 'Phosphorylation', # miss
    'Phosphoramido_Nitrile': 'Phosphorylation', # miss
    # page s-13
    'Phosphoric_Acid': 'Phosphorylation', # miss
    'Phosphorothioate_Ester': 'Phosphorylation', # miss
    'Propynamide': 'Nucleophilic Addition to a Triple Bond',
    'Purine': 'Nucleophilic Aromatic Substitution',
    'Sulfonate_Ester': 'Sulfonylation', # miss
    'Sulfonic_Acid': 'Sulfonylation', # miss
    'Sulfonyl_Halide': 'Sulfonylation',
    # page s-14
    'Sulfoxide': 'Sulfonylation', # miss
    'Thiirane': 'Ring Opening',
    'Thioester': 'Nucleophilic Acyl Substitution', # miss
    'Thioether': 'Disulfide Formation', # miss
    'Thiol': 'Disulfide Formation',
    'Thiophosphonate_Ester': 'Phosphorylation', # miss
    'Thiosulfonate': 'Disulfide Formation',
    'Ubiquitin_Aldehyde': 'Hemi(thio)acetalization', # miss
    # page s-15
    'Urea': 'Nucleophilic Acyl Substitution', # miss
    'Uridine': 'Nucleophilic Addition to a Double Bond', # not sure and miss
    'Vinyl_Sulfonamide': 'Nucleophilic Addition to a Double Bond', # miss
    'Vinyl_Sulfonate': 'Nucleophilic Addition to a Double Bond', # miss
    'Vinyl_Sulfone': 'Nucleophilic Addition to a Double Bond', # miss
    'α,β-unsaturated_Carbonyl': 'Nucleophilic Addition to a Double Bond', # miss
    # page s-16
    'α-acyloxymethyl_Ketone': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-amino_Carbonyl': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-aryloxymethyl_Ketone': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-cyanovinyl_Carbonyl': 'Michael Addition',
    'α-diazomethyl_Ketone': 'Nucleophilic Aliphatic Subsititution', # miss
    # page s-17
    'α-haloamide': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-halocarboxylate': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-halocarboxylic_Acid': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-halomethyl_Ketone': 'Nucleophilic Aliphatic Subsititution', # miss
    'α-hydroxy_Sulfonic_Acid': 'Hemi(thio)acetalization',
    'α-ketoamide': 'Nucleophilic Acyl Substitution', # miss
    # page s-18
    'α-pyrone': 'Lactone Addition',
    'β-lactam': 'Beta-Lactam Addition',
    'β-lactone': 'Lactone Addition',
    'β-sulfonylvinyl_Carbonyl': 'Michael Addition',
    'β-sulfonylvinyl_Nitrile': 'Michael Addition',
    'β-sultam': 'Ring Opening',
    # page s-19
    'γ-lactam': 'Ring Opening',
    'γ-lactone': 'Lactone Addition',
    
    # Unclassified
    'Unclassified': 'Unclassified'
}


if __name__=='__main__':
    CovBinderInPDB_data_info_filename = 'data/covbinderInPDB/CovBinderInPDB_2022Q4_AllRecords.csv'
    
    orig_df = pd.read_csv(CovBinderInPDB_data_info_filename)
    # get the pdb_id and warhead_name columns
    covbinder_pdb2mechanism = orig_df[['pdb_id', 'warhead_name']]
    # drop duplicates
    covbinder_pdb2mechanism = covbinder_pdb2mechanism.drop_duplicates()
    
    # map warhead names to mechanisms
    covbinder_pdb2mechanism['mechanism'] = covbinder_pdb2mechanism['warhead_name'].map(warhead2mechanism)
    
    # export to csv
    covbinder_pdb2mechanism.to_csv('./data/auxiliary/covbinderInPDB_pdb2mechanism.csv', index=False)