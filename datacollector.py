import os
import sys
import urllib.request
import json
import csv


class DataCollector:
    def __init__(self, ponzi_filename, non_ponzi_filename):
        self.ponzi_filename = ponzi_filename
        self.non_ponzi_filename = non_ponzi_filename
        self.ponzi_contract_list = []
        self.non_ponzi_contract_list = []
        self.baseURL = "http://ibasetest.inpluslab.com/scamedb/contract_download/static/file/_Txs/ContractNormalTx/"

        self.get_ponzi_contract_list()
        self.get_nonponzi_contract_list()

        self.queryPonziUrlAndSaveToFiles()
        self.queryNonPonziUrlAndSaveToFiles()

        self.api_internal_tx_list = "http://api.etherscan.io/api?module=account&action=txlistinternal&address="
        self.query_ponzi_internal_tx_save_to_file()
        self.query_nonponzi_internal_tx_save_to_file()

    def get_ponzi_contract_list(self):
        with open(self.ponzi_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # line# = 0 -> header
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    # print(f'\t{row[0]}, {row[1]}, {row[2]}.')
                    self.ponzi_contract_list.append(row[1])
                    line_count += 1

            print(f'Processed {line_count} lines.')
            print(self.ponzi_contract_list)

    def get_nonponzi_contract_list(self):
        with open(self.non_ponzi_filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:  # line# = 0 -> header
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    # print(f'\t{row[0]}, {row[1]}, {row[2]}.')
                    self.non_ponzi_contract_list.append(row[1])
                    line_count += 1

            # print(f'Processed {line_count} lines.')
            # print(self.ponzi_contract_list)

    def readUrlGetJson(self, contractAddress):
        contractUrl = self.baseURL + contractAddress + ".json"
        # print(contractUrl)
        results = []
        with urllib.request.urlopen(contractUrl) as url:
            data = json.loads(url.read().decode())
            return data['result']

    def read_api_get_json(self, contractAddress):
        api_url = self.api_internal_tx_list + contractAddress

        with urllib.request.urlopen(api_url) as api_url:
            data = json.loads(api_url.read().decode())
            return data['result']

    def queryPonziUrlAndSaveToFiles(self):
        print('=== Downloading ponzi transaction files...')
        counter = 0
        for contratAddress in self.ponzi_contract_list:
            fileNameToSave = './data/transactions/ponzi/' + contratAddress + '.csv'
            if not os.path.exists(fileNameToSave):
                with open(fileNameToSave, 'w') as f:
                    f.write(" ")

            results = self.readUrlGetJson(contratAddress)
            fieldnames = ['blockNumber', 'blockHash', 'timeStamp', 'hash', 'nonce', 'transactionIndex', 'from', 'to', 'value',
                          'gas', 'gasPrice', 'input', 'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'isError']
            with open(fileNameToSave, mode='w+') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    result['input'] = ''
                    # print(result)
                    writer.writerow(result)
            counter += 1
            if counter % 500 == 0 and counter > 0:
                print('{0} transactions have downloaded...'.format(counter))
        print('ponzi transactions downloading is over.')

    def queryNonPonziUrlAndSaveToFiles(self):
        print('=== Downloading nonponzi transaction files...')
        counter = 0
        for contratAddress in self.non_ponzi_contract_list:
            fileNameToSave = './data/transactions/nonponzi/' + contratAddress + '.csv'
            if not os.path.exists(fileNameToSave):
                with open(fileNameToSave, 'w') as f:
                    f.write(" ")

            results = self.readUrlGetJson(contratAddress)
            fieldnames = ['blockNumber', 'blockHash', 'timeStamp', 'hash', 'nonce', 'transactionIndex', 'from', 'to', 'value',
                          'gas', 'gasPrice', 'input', 'contractAddress', 'cumulativeGasUsed', 'gasUsed', 'confirmations', 'isError']
            with open(fileNameToSave, mode='w+') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    result['input'] = ''
                    # print(result)
                    writer.writerow(result)
            counter += 1
            if counter % 500 == 0 and counter > 0:
                print('{0} transactions have downloaded...'.format(counter))
        print('nonponzi transactions downloading is over.')

    def query_ponzi_internal_tx_save_to_file(self):
        print('=== Downloading ponzi internal transaction files...')
        counter = 0
        for contratAddress in self.ponzi_contract_list:
            tx_list = self.read_api_get_json(contratAddress)
            fileNameToSave = './data/internal_transactions/ponzi/' + contratAddress + '.csv'
            if not os.path.exists(fileNameToSave):
                with open(fileNameToSave, 'w') as f:
                    f.write(" ")

            with open(fileNameToSave, mode='w+') as csv_file:
                fieldnames = ['blockNumber', 'timeStamp', 'hash', 'from', 'to', 'value',
                              'contractAddress', 'input', 'type', 'gas', 'gasUsed', 'traceId', 'isError', 'errCode']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for tx in tx_list:
                    tx['input'] = ''
                    writer.writerow(tx)
            counter += 1
            if counter % 500 == 0 and counter > 0:
                print('{0} internal transactions have downloaded...'.format(counter))
        print('ponzi internal transactions downloading is over.')

    def query_nonponzi_internal_tx_save_to_file(self):
        print('=== Downloading nonponzi internal transaction files...')
        counter = 0
        for contratAddress in self.non_ponzi_contract_list:
            tx_list = self.read_api_get_json(contratAddress)
            fileNameToSave = './data/internal_transactions/nonponzi/' + contratAddress + '.csv'
            if not os.path.exists(fileNameToSave):
                with open(fileNameToSave, 'w') as f:
                    f.write(" ")

            with open(fileNameToSave, mode='w+') as csv_file:
                fieldnames = ['blockNumber', 'timeStamp', 'hash', 'from', 'to', 'value',
                              'contractAddress', 'input', 'type', 'gas', 'gasUsed', 'traceId', 'isError', 'errCode']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                if tx_list is not None:
                    for tx in tx_list:
                        tx['input'] = ''
                        writer.writerow(tx)
            counter += 1
            if counter % 500 == 0 and counter > 0:
                print('{0} internal transactions have downloaded...'.format(counter))
        print('nonponzi internal transactions downloading is over.')


if __name__ == '__main__':
    # The two files are generated by splitting the flag.csv file
    datacollector = DataCollector(
        './ponziContracts.csv', './non_ponziContracts.csv')
