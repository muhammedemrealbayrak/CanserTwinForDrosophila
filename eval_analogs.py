
import sys, os
sys.path.insert(0, os.path.abspath('.'))

from pipeline.pubchem_connector import PubChemConnector
from denovo_ai.fitness_evaluator import MolecularFitnessEvaluator
from platform_engine import DrosophilaInSilicoPlatform
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

pubchem = PubChemConnector()

test_molecules = [
    ('Nicotine (Referans)', 'CN1CCC[C@H]1c2cccnc2'),
    ('Analog 1: Fluoronicotinil-Asetamid (F-NAc)', 'NC(=O)CN1CCC[C@H]1c2cncc(F)c2'),
    ('Analog 2: Bis-Floro Asetil-Nornikotin (F2-AcN)', 'Fc1cncc(c1)[C@@H]2CCCN2C(=O)CF'),
    ('Analog 3: Metil Karbamat Nornikotin (MCN)', 'COC(=O)N1CCC[C@H]1c2cccnc2'),
    ('De Novo AI Sampiyon (Optimize)', 'CC1=NC=C(C=C1)CCN(C)C(=O)CF')
]

print('| ' + 'Molekul Adi'.ljust(38) + ' | ' + 'MW'.ljust(6) + ' | ' + 'LogP'.ljust(5) + ' | ' + 'TPSA'.ljust(6) + ' | ' + 'QED'.ljust(5) + ' | ' + 'Kd(uM)'.ljust(6) + ' | ' + 'Tetikleme'.ljust(9) + ' | ' + 'Tumor(%)'.ljust(8) + ' | ' + 'Toksisite(%)'.ljust(13) + ' | ' + 'Fitness'.ljust(7) + ' |')
print('|' + '-'*40 + '|' + '-'*8 + '|' + '-'*7 + '|' + '-'*8 + '|' + '-'*7 + '|' + '-'*8 + '|' + '-'*11 + '|' + '-'*10 + '|' + '-'*15 + '|' + '-'*9 + '|')

for name, smiles in test_molecules:
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    qed = QED.qed(mol)
    
    prof = pubchem.parse_molecule(smiles)
    engine = DrosophilaInSilicoPlatform(active_compound_smiles_or_name=smiles, initial_tumor_burden=150)
    res = engine.run_benchmark(duration_seconds=600.0)
    
    print(f'| {name:<38} | {mw:6.1f} | {logp:5.2f} | {tpsa:6.1f} | {qed:5.2f} | {prof.kd_micromolar:6.3f} | {res.time_to_trigger_s:7.1f} s | %{res.tumor_clearance_pct:6.1f} | %{res.systemic_toxicity_pct:11.1f} | {res.multiobjective_fitness_score:5.1f}   |')
