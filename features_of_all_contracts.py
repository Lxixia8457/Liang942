import os
import csv
import sys
import numpy as np
from opcodefeature_extraction import ContractFeature

class ExtractFeatureOfAllContract(object):
    def __init__(self):
        self.ponzi_file = './data/ponziContracts.csv'
        self.non_ponzi_file = './data/non_ponziContracts.csv'
        self.feature_save_file = './data/features.csv'
        self.feature2_save_file = './data/features2.csv'
        self.fieldnames = ['address', 'kr', 'bal', 'n_inv', 'n_pay', 'pr', 'n_max', 'd_ind', 'tra_inv', 'tra_pay', 'n_invor', 'n_payer', 'std_pay']
        self.fieldnames2 = ['GASLIMIT', 'EXP', 'CALLDATALOAD', 'SLOAD', 'CALLER', 'LT', 'GAS', 'MOD', 'MSTORE', 'ponzi']

        self.initialize_files()

        self.extract_features()
        
        print()

    def initialize_files(self):
        # extract feature of opcode
        if os.path.exists(self.feature2_save_file):
            os.remove(self.feature2_save_file)
        if not os.path.exists(self.feature2_save_file):
            with open(self.feature2_save_file, 'w') as f2:
                writer = csv.DictWriter(f2, fieldnames=self.fieldnames + self.fieldnames2)
                writer.writeheader()

    def extract_features(self):
        with open(self.ponzi_file, 'r') as ponzi_csv:
            csv_reader = csv.reader(ponzi_csv, delimiter=',')
            line_count = 0
            for row in csv_reader:
                features2 = {}
                if line_count > 0:
                    contractAddress = row[1]
                    if os.path.exists('./data/contracts/ponzi/' + contractAddress + '.txt'):
                        contractFeature = ContractFeature(contractAddress, 'ponzi')

                        features2['address'] = contractAddress
                        features2['kr'] = contractFeature.kr
                        features2['bal'] = contractFeature.bal
                        features2['n_inv'] = contractFeature.n_inv
                        features2['n_pay'] = contractFeature.n_pay
                        features2['pr'] = contractFeature.pr
                        features2['n_max'] = contractFeature.n_max
                        features2['d_ind'] = contractFeature.d_ind
                        features2['tra_inv'] = contractFeature.tra_inv
                        features2['tra_pay'] = contractFeature.tra_pay
                        features2['n_invor'] = contractFeature.n_invor
                        features2['n_payer'] = contractFeature.n_payer
                        features2['std_pay'] = contractFeature.std_pay
                        features2['GASLIMIT'] = contractFeature.action_ratio['GASLIMIT']
                        features2['EXP'] = contractFeature.action_ratio['EXP']
                        features2['CALLDATALOAD'] = contractFeature.action_ratio['CALLDATALOAD']
                        features2['SLOAD'] = contractFeature.action_ratio['SLOAD']
                        features2['CALLER'] = contractFeature.action_ratio['CALLER']
                        features2['LT'] = contractFeature.action_ratio['LT']
                        features2['GAS'] = contractFeature.action_ratio['GAS']
                        features2['MOD'] = contractFeature.action_ratio['MOD']
                        features2['MSTORE'] = contractFeature.action_ratio['MSTORE']
                        features2['ponzi'] = 1

                        with open(self.feature2_save_file, 'a') as feat2_file_csv:
                            writer = csv.DictWriter(feat2_file_csv, fieldnames=self.fieldnames + self.fieldnames2)
                            writer.writerow(features2)
                features2 = {}
                line_count += 1

        with open(self.non_ponzi_file, 'r') as nonponzi_csv:
            csv_reader = csv.reader(nonponzi_csv, delimiter=',')
            line_count = 0
            for row in csv_reader:
                features2 = {}
                if line_count > 0:
                    contractAddress = row[1]
                    if os.path.exists('./data/contracts/nonponzi/' + contractAddress + '.txt'):
                        contractFeature = ContractFeature(contractAddress, 'nonponzi')

                        features2['address'] = contractAddress
                        features2['kr'] = contractFeature.kr
                        features2['bal'] = contractFeature.bal
                        features2['n_inv'] = contractFeature.n_inv
                        features2['n_pay'] = contractFeature.n_pay
                        features2['pr'] = contractFeature.pr
                        features2['n_max'] = contractFeature.n_max
                        features2['d_ind'] = contractFeature.d_ind
                        features2['tra_inv'] = contractFeature.tra_inv
                        features2['tra_pay'] = contractFeature.tra_pay
                        features2['n_invor'] = contractFeature.n_invor
                        features2['n_payer'] = contractFeature.n_payer
                        features2['std_pay'] = contractFeature.std_pay
                        features2['GASLIMIT'] = contractFeature.action_ratio['GASLIMIT']
                        features2['EXP'] = contractFeature.action_ratio['EXP']
                        features2['CALLDATALOAD'] = contractFeature.action_ratio['CALLDATALOAD']
                        features2['SLOAD'] = contractFeature.action_ratio['SLOAD']
                        features2['CALLER'] = contractFeature.action_ratio['CALLER']
                        features2['LT'] = contractFeature.action_ratio['LT']
                        features2['GAS'] = contractFeature.action_ratio['GAS']
                        features2['MOD'] = contractFeature.action_ratio['MOD']
                        features2['MSTORE'] = contractFeature.action_ratio['MSTORE']
                        features2['ponzi'] = 0

                        with open(self.feature2_save_file, 'a') as feat2_file_csv:
                            writer = csv.DictWriter(feat2_file_csv, fieldnames=self.fieldnames + self.fieldnames2)
                            writer.writerow(features2)
                features2 = {}
                line_count += 1
        
        print(features2)
